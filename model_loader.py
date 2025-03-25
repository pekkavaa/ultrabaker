import pygltflib
from pygltflib import GLTF2
import struct
import numpy as np
from pathlib import Path
import urllib.parse
import base64

def accessor_to_size_format(accessor):
    format = '<'
    types = {
        pygltflib.FLOAT: 'f',
        pygltflib.UNSIGNED_INT: 'I',
        pygltflib.UNSIGNED_SHORT: 'H',
    }
    sizes = {
        pygltflib.FLOAT: 4,
        pygltflib.UNSIGNED_INT: 4,
        pygltflib.UNSIGNED_SHORT: 2,
    }

    bpv = sizes[accessor.componentType]

    if accessor.type == 'SCALAR':
        format += types[accessor.componentType]
    if accessor.type == 'VEC2':
        format += types[accessor.componentType] * 2
        bpv *= 2
    elif accessor.type == 'VEC3':
        format += types[accessor.componentType] * 3
        bpv *= 3
    else:
        raise RuntimeError(f"Unknown accessor type {accessor.type}")

    return bpv, format
    
def load_indices(gltf, mesh):
    # triangles_accessor = gltf.accessors[gltf.meshes[0].primitives[0].indices]
    primitive = mesh.primitives[0]
    
    # bpv, format = attribute_to_size_format(attribute_name)
    # attribute = getattr(primitive.attributes, attribute_name)

    # get the binary data for this mesh primitive from the buffer
    accessor = gltf.accessors[primitive.indices]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)

    values = []

    if accessor.componentType == pygltflib.UNSIGNED_INT:
        bpv = 4
        format = "<I"
    elif accessor.componentType == pygltflib.UNSIGNED_SHORT:
        bpv = 2
        format = "<H"
    else:
        raise RuntimeError(f"Unknown component type {accessor.componentType}")


    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*bpv
        d = data[index:index+bpv]
        v = struct.unpack(format, d)
        if len(v) == 1:
            v = v[0]  # convert single tuples to values
        values.append(v)
    
    return np.array(values)


def load_attribute(gltf, mesh, attribute_name):
    assert len(mesh.primitives) == 1
    primitive = mesh.primitives[0]
    
    attribute = getattr(primitive.attributes, attribute_name)
    accessor = gltf.accessors[attribute]

    bpv, format = accessor_to_size_format(accessor)

    # get the binary data for this mesh primitive from the buffer
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)

    vertices = []

    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*bpv  # the location in the buffer of this vertex
        d = data[index:index+bpv]  # the vertex data
        v = struct.unpack(format, d)   # convert to floats
        vertices.append(v)
        # print(i, v)
    
    return np.array(vertices)

def extract_pos_uvs_tris_img(gltf, filename):
    mesh = gltf.meshes[gltf.scenes[gltf.scene].nodes[0]]

    inds = load_indices(gltf, mesh)
    uvs = load_attribute(gltf, mesh, 'TEXCOORD_0')
    positions = load_attribute(gltf, mesh, 'POSITION')

    img_path = urllib.parse.unquote(gltf.images[0].uri)
    img_rel_path = Path(filename).with_name(img_path)
    print(img_rel_path)

    import skimage.io
    img = skimage.io.imread(img_rel_path)
    img = img.astype(np.float32) / 255.0

    if img.shape[2] == 4:
        print("Alpha channel ignored: Using only RGB channels of an RGBA image.")
        img = img[...,:3]
    
    tris = []
    assert inds.shape[0]//3, "Only triangle faces expected"
    inds = inds.reshape(-1,3)
    # Convert np.array to a list of tuples process.py expects
    for i in range(inds.shape[0]):
        a, b, c = inds[i]
        tris.append((a,b,c))

    # Convert UVs to pixels
    uvs[:,0] *= img.shape[1]
    uvs[:,1] *= img.shape[0]

    return positions, uvs, tris, img

base64_start = "data:application/octet-stream;base64,"

def load_uri_as_bytes(uri, model_filename):
    """Decodes base64 or loads binary data from disk, depending on URI format."""
    if uri.startswith(base64_start):
        return base64.b64decode(uri)
    else:
        rel_path = Path(urllib.parse.unquote(model_filename)).with_name(uri)
        print(f"Loading data file {rel_path}")
        with open(rel_path, 'rb') as f:
            return f.read()


def add_vertex_colors(gltf, colors, filename):
    lightmap_image_index = 0
    lightmap_texture_index = 0
    lightmap_material_index = 0
    # lightmap_texcoord_index = 0
    # del gltf.images[lightmap_image_index]
    # del gltf.textures[lightmap_texture_index]
    gltf.materials[lightmap_material_index].pbrMetallicRoughness.baseColorTexture = None

    colors_rgbx = np.zeros((colors.shape[0], 4), dtype=colors.dtype)
    colors_rgbx[:,0:3] = colors
    colors_bytes = colors_rgbx.tobytes()

    data = load_uri_as_bytes(gltf.buffers[0].uri, filename)
    assert len(data)%4==0, "existing data should be 4 byte aligned already"
    # data += b'\00' * (len(data)%4) # Pad data if it isn't aligned yet

    color_offset = len(data)
    data += colors_bytes
    gltf.buffers[0].uri = base64_start + base64.encodebytes(data).decode('utf-8')

    buf_idx = 0
    # Add a bufferview with a stride=4 so that accesses are 4-byte aligned. Required by the GLTF spec.
    gltf.bufferViews.append(pygltflib.BufferView(buffer=buf_idx, byteOffset=color_offset, byteLength = len(colors_bytes), byteStride=4))
    view_idx = len(gltf.bufferViews)-1
    # Add an accessor
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=view_idx, byteOffset=0, componentType=pygltflib.UNSIGNED_BYTE, normalized=True, count=colors_rgbx.shape[0], type='VEC4'))
    accessor_idx = len(gltf.accessors)-1
    # Point to accessor index in mesh primitive attributes
    gltf.meshes[0].primitives[0].attributes.COLOR_0 = accessor_idx

    return gltf

def save_big_endian_dump(path: str, positions, tris, colors):
    """
    Saves a model in a simple binary format. It's big-endian so that
    it's easy to directly load on the console.
    """
    from struct import pack

    assert positions.shape[0] == colors.shape[0]
    assert colors.dtype == np.uint8

    if colors.shape[1] == 3:
        print("Adding an alpha=255 channel to colors")
        colors_temp = np.full((colors.shape[0], 4), 255, dtype=colors.dtype)
        colors_temp[:,:3] = colors
        colors = colors_temp

    # compute sizes of fields
    num_verts = positions.shape[0]
    num_inds = len(tris)*3
    for tri in tris:
        assert len(tri) == 3, "Only triangle faces supported in binary export"
    
    vertex_array_byte_size = num_verts * 4 * 4
    index_array_byte_size = num_inds * 2

    with open(path, 'wb') as f:
        # Write a header
        f.write(b'BINM')
        f.write(pack('>I', 20230603)) # a version number
        cur = f.tell()
        header_size = 2 * 2 * 4

        vertex_start = header_size + cur
        index_start = header_size + cur + vertex_array_byte_size

        f.write(pack('>II', num_verts, num_inds))
        f.write(pack('>II', vertex_start, index_start))

        # Write (X,Y,Z,RGBA) vertices
        for i in range(num_verts):
            x, y, z = positions[i]
            f.write(pack('>fff', x, y, z))
            r, g, b, a = colors[i]
            rgba = (r << 24) | (g << 16) | (b << 8) | a
            f.write(pack('>I', rgba))
        
        # Write the triangle list as 16-bit indices
        for a, b, c in tris:
            f.write(pack('>HHH', a, b, c))


if __name__ == "__main__":
    filename = 'models/simplescene_joined_vertex_colors.gltf'
    gltf = GLTF2().load(filename)
    positions, uvs, tris, img = extract_pos_uvs_tris_img(gltf, filename)
    print('UV shape:', uvs.shape)
    print('tris shape', len(tris))
    print('img shape:', img.shape)

    colors = np.random.uniform(0,256,size=(uvs.shape[0],3))
    gltf = add_vertex_colors(gltf, colors, filename)

    from pygltflib.validator import validate, summary
    validate(gltf)
    summary(gltf)

    gltf.save_binary('output/baked_random.glb')
    colors_bytes = np.clip(colors, 0, 255).astype(np.uint8)
    save_big_endian_dump('output/baked_random.binm', positions, tris, colors_bytes)

