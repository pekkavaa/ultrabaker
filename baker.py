from argparse import ArgumentParser
from pathlib import Path
import pygltflib
from pygltflib import GLTF2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import time

import model_loader

parser = ArgumentParser(description="Bakes GLTF models with lightmaps to vertex colors")
parser.add_argument("input", help="Path to input .GLTF model")
parser.add_argument("output", default="baked.glb", nargs='?', help="Name of output model.")
parser.add_argument("--smoothing", type=float, default=20, help="Color smoothing value in range [0,100]")
parser.add_argument("--show", action='store_true', help="Show the baking result visualization at the end.")
args = parser.parse_args()
print(args)


np.random.seed(123)


def cross2d(a,b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def rasterize_naive(img_shape, v0, v1, v2):
    p_rows, p_cols = np.indices(img_shape)
    # 'p' contains each pixel coordinate as float (x,y)
    p = np.stack([p_cols, p_rows], axis=2)
    a = p - v0
    b = p - v1
    c = p - v2

    barycentrics = np.zeros((*img_shape[:2], 3))
    barycentrics[...,0] = cross2d(c, b)
    barycentrics[...,1] = cross2d(a, c)
    barycentrics[...,2] = cross2d(b, a)

    mask3 = barycentrics > 0
    mask = mask3[...,0] & mask3[...,1] & mask3[...,2]

    area2x = cross2d(v2 - v0, v1 - v0)
    return mask, barycentrics / area2x


def rasterize(img_shape, v0, v1, v2):
    """
    A Pineda-style triangle rasterizer.
    Somewhat sloppy since it's all floats and doesn't respect proper fill rules.
    """

    # Compute triangle's bounding box and clamp its bounds to image size
    mins = np.floor(np.array([v0,v1,v2]).min(axis=0)).astype(np.int32)
    maxs = np.ceil(np.array([v0,v1,v2]).max(axis=0)).astype(np.int32)
    mins = np.maximum((0,0), mins)
    maxs = np.minimum([img_shape[1]-1, img_shape[0]-1], maxs)

    # Compute the shape of the bounding box
    # Shape convention is (rows, cols) but bounds were computed from (x,y) vertices so we swap elements here
    shape = maxs[[1,0]] - mins[[1,0]] 

    # Bias vertices to top-left corner of the bounding box
    v0 = v0 - mins
    v1 = v1 - mins
    v2 = v2 - mins

    # Store local coordinates of each pixel
    # The array 'p' contains each pixel coordinate as float (x,y).
    p_rows, p_cols = np.indices(shape)
    p = np.stack([p_cols, p_rows], axis=2)

    # Vertex-to-pixel vectors to be used in edge equations below
    a = p - v0
    b = p - v1
    c = p - v2

    # Compute unnormalized barycentric coordinates for each vertex
    edge_equations = np.zeros((*shape, 3))
    edge_equations[...,0] = cross2d(c, b)
    edge_equations[...,1] = cross2d(a, c)
    edge_equations[...,2] = cross2d(b, a)

    # Pixels have a non-negative edge equation for each vertex
    # Using greater-or-equal comparison here makes the rasterizer a bit conservative
    # so neighboring triangles will fill some pixels twice.
    mask3 = edge_equations >= 0
    mask = mask3[...,0] & mask3[...,1] & mask3[...,2]

    # Create full mask and barycentric arrays and assign to full-sized arrays
    full_mask = np.zeros(img_shape[:2], dtype=bool)
    full_barycentrics = np.zeros((*img_shape[:2], 3))

    full_mask[mins[1]:maxs[1], mins[0]:maxs[0]] = mask

    # Scale edge distances to actual normalized barycentric coordinates
    area2x = cross2d(v2 - v0, v1 - v0)
    full_barycentrics[mins[1]:maxs[1], mins[0]:maxs[0]] = edge_equations / area2x
    return full_mask, full_barycentrics


def draw_triangles(vert_coords, vert_values, tris, backfacing, img):
    for tri_idx, tri in enumerate(tris):
        v0, v1, v2 = np.take(vert_coords, tri, axis=0)
        f0, f1, f2 = np.take(vert_values, tri, axis=0)
        # print("tri_idx", tri_idx, "tri", tri, "v0", v0, "v1", v1)
        if backfacing[tri_idx]:
            mask, weights = rasterize(img.shape, v0, v1, v2)
        else:
            mask, weights = rasterize(img.shape, v0, v2, v1)
        result = weights[mask,0:1] * f0 + weights[mask,1:2] * f1 + weights[mask,2:3] * f2
        img[mask] = result


def generate_2d_mesh(img, N_verts):
    if False:
        # Vertex positions
        verts = [
            np.array([24,12]),
            np.array([45,48]),
            np.array([81,22]),
            np.array([60,5]),
        ]

        # Vertex f(p) signals (baseline)
        vert_fs_groundtruth = [0.2, 0.8, 1.0, 0.0]
    else:
        from scipy import ndimage as ndi
        sigma = max(*img.shape)/50.0
        # print('sigma:', sigma)
        img_blurred = ndi.gaussian_filter(img, sigma)
        img_edges = np.abs(img - img_blurred)
        img_edges = ndi.gaussian_filter(img_edges, sigma/2).astype(np.float64)
        img_edges = img_edges**2
        edge_probas = img_edges / np.sum(img_edges)

        img_rows, img_cols = np.indices(img.shape)
        # 'p' contains each pixel coordinate as float (x,y)
        img_coords = np.stack([img_cols, img_rows], axis=2)
        vids = np.random.choice(img_coords.size//2, N_verts, p=edge_probas.flatten(), replace=False)
        verts = img_coords.reshape(-1,2)[vids].astype(np.float32)

        # fig, ax = plt.subplots(1, figsize=(8,12))
        # ax.imshow(img_edges)
        # plt.show()

    # Make sure image corners always have vertices available
    verts[0] = np.array([0,0])
    verts[1] = np.array([img.shape[1]-1,0])
    verts[2] = np.array([0,img.shape[0]-1])
    verts[3] = np.array([img.shape[1]-1,img.shape[0]-1])

    print('Triangulating')
    from scipy.spatial import Delaunay
    mesh_result = Delaunay(verts, qhull_options="Qbb Qc Qz Q12") # "QJ" guarantees all points are used but generates extra vertices?

    # fig, ax = plt.subplots(1, 2, figsize=(12,8))
    # verts_array = np.array(verts)
    # ax.flatten()[0].imshow(edge_probas)
    # ax.flatten()[1].imshow(edge_probas)
    # ax.flatten()[1].triplot(verts_array[:,0], verts_array[:,1], mesh_result.simplices, color='white')
    # plt.show()

    tris = []
    for tri_idx in range(mesh_result.simplices.shape[0]):
        i0, i1, i2 = mesh_result.simplices[tri_idx, :]
        v0, v1, v2 = np.take(verts, [i0, i1, i2], axis=0)
        area2x = cross2d(v2 - v0, v1 - v0)
        if abs(area2x) < 1:
            # skip if triangle area is small
            continue
        if area2x > 0:
            tris.append((i0, i1, i2))
        else:
            tris.append((i0, i2, i1)) # flip winding to counter-clockwise
    
    return verts, tris


if False:
    print('Loading image')

    # img = skimage.io.imread('baker/monalisa.png')
    # img = skimage.io.imread('baker/sky_greyscale.png')
    img = skimage.io.imread('baker/sky_rgb.png')
    img = img.astype(np.float32) / 255.0
    N_verts = 100
    uvs, tris = generate_2d_mesh(img[...,1], 200) # Generate mesh with green channel
else:
    filename = args.input
    print(f"Loading {filename}")
    gltf = GLTF2().load(filename)
    positions, normals, uvs, raw_tris, img = model_loader.extract_pos_uvs_tris_img(gltf, filename)


# img2 = np.zeros_like(img)
# xtest = np.random.uniform(0, 1, size=(N,3))
# draw_triangles(uvs, xtest, tris, img2)

# fig, ax = plt.subplots(2, 1, figsize=(12,8))
# # verts_array = np.array(uvs)
# ax.flatten()[0].imshow(img)
# ax.flatten()[1].imshow(img2)
# plt.show()

# Input may have disjoint UV coordinates in the light map but we'll merge those vertices
# if they have the same position and normal.
# Therefore two triangles are considered to share an edge if the edge has the same
# position and normal.

# Transform original triangles to a {old-edge -> new-edge} mapping.

backfacing_raw: list[bool] = []

for tri in raw_tris:
    v0, v1, v2 = np.take(uvs, tri, axis=0)
    area = cross2d(v1 - v0, v2 - v0) / 2
    backfacing_raw.append(area < 0)

deduplicate = True

if deduplicate:
    pos_to_id: dict[tuple, int] = dict()
    new_tris = []
    new_vertex_id_to_old = dict()
    old_vertex_id_to_new = dict()

    # for i in range(positions.shape[0]):
    #     p = positions[i]
    #     for j in range(i+1, positions.shape[0]):
    #         k = positions[j]
    #         print(i,j,"match")
    #         # assert(not np.all(p == k))

    def get_location_key(p, n):
        x,y,z = p
        nx, ny, nz = n
        return (float(x), float(y), float(z), float(nx), float(ny), float(nz))


    for tri_idx, (a, b, c) in enumerate(raw_tris):
        new_tri = []
        for i0 in [a,b,c]:
            key = get_location_key(positions[i0], normals[i0])
            
            if key in pos_to_id:
                i1 = pos_to_id[key]
            else:
                pos_to_id[key] = i0
                i1 = i0
            
            new_vertex_id_to_old[i1] = i0
            old_vertex_id_to_new[i0] = i1
            new_tri.append(i1)
        
        new_tris.append(tuple(new_tri))
            

    print("pos_to_id:")
    for key, value in pos_to_id.items():
        print(key, value)

    for tri in new_tris:
        for i0 in tri:
            assert(i0 in new_vertex_id_to_old)

    for tri_idx, ((a, b, c), (a2, b2, c2)) in enumerate(zip(raw_tris, new_tris)):
        print(f"[{tri_idx}] ({a}, {b}, {c}) -> ({a2}, {b2}, {c2})")

    print("new_tris:")
    # print(new_tris)
    print(f"new_tris length: {len(new_tris)} vs {len(raw_tris)} of old")
    print()


    tris = new_tris
else:
    tris = raw_tris

N = len(uvs)



# Go over each original edge and check if it's already traversed on the other side.
# If it is, emit the saved indices to the edge array.
# If not, add it both ways with new vertex indices.

# Recreate the triangles array with new edges.
# The triangles array is now a series of compressed vertex indices.
# Fit colors.
# Finally, use the old-edge -> new-edge mapping to create a longer, duplicated color result.

print('Finding vertex neighbors')

edge_tris = {}
vertex_tris = [[] for i in range(N)]
vertex_neighbors = [set() for i in range(N)]


for tri_idx, (a, b, c) in enumerate(tris):
    # If two triangles with the same winding share an edge, then the other
    # side will have it as (i, j) and the other as (j, i). The 'edge_tris' array
    # should contain *all* triangles incident to an edge used as a key, so we
    # add the triangle to both ways below.
    for i, j in [(a,b), (b,c), (c,a)]:
        edge_tris.setdefault((i,j), []).append(tri_idx)
        edge_tris.setdefault((j,i), []).append(tri_idx)
        vertex_neighbors[i].add(j)
        vertex_neighbors[j].add(i)

    vertex_tris[a].append(tri_idx)
    vertex_tris[b].append(tri_idx)
    vertex_tris[c].append(tri_idx)

validate_neighbors = True

if validate_neighbors:
    for i, neighs in enumerate(vertex_neighbors):
        my_tri_indices = vertex_tris[i]
        for j in neighs:
            same_tri = -1
            for tri_idx in my_tri_indices:
                tri = tris[tri_idx]
                if i in tri and j in tri:
                    same_tri = tri_idx
                    break
            assert same_tri > -1, "Bug: Each neighbor in 'vertex_neighbors' must belong to some same triangle"
        


tri_areas = []
backfacing: list[bool] = []
num_negative_area = 0

for tri in tris:
    v0, v1, v2 = np.take(uvs, tri, axis=0)
    area = cross2d(v1 - v0, v2 - v0) / 2
    backfacing.append(area < 0)
    if area < 0:
        num_negative_area += 1
        area = -area
    tri_areas.append(area)

ratio_negative = num_negative_area / len(tris)
if ratio_negative > 0.1:
    print(f"Warning: {ratio_negative*100:.1f} % of triangles have a negative UV surface area!")

A = np.zeros((N,N))

for i in range(N):
    for tri_idx in vertex_tris[i]:
        A[i,i] += tri_areas[tri_idx] / 6

# print("Edges")
for (i, j), tri_inds in edge_tris.items():
    # print(f"({i}, {j}) = {tri_inds}")
    for tri_idx in tri_inds:
        A[i, j] += tri_areas[tri_idx] / 12


print('Building regularization matrix R')

# Build a Laplacian matrix
R=np.zeros((N,N))
for i in range(N):
    num = 0
    for j in vertex_neighbors[i]:
        R[i,j] = -1
        num += 1
    R[i,i] = num

assert np.abs(R - R.T).max() < 1e-6, "R should be symmetric"

print('R:')
print(R)

print('Building the system matrix A')

build_start = time.time()
verify_system_matrix = False

if verify_system_matrix:
    import scipy.linalg
    A_eigenvalues, _ = scipy.linalg.eig(A)
    assert (A == A.T).all(), "'A' should be symmetric"
    assert (A_eigenvalues >= 0).all(), "'A' should have positive eigenvalues because it's a positive definite matrix"

print('Building the target vector b')

b = np.zeros((N,3))

for i in tqdm(range(N)):
    # Find triangles that neighbor this triangle.
    neighs = vertex_tris[i]

    # Sample a linear "hat" function in the pixel grid that is 1 directly on the
    # the vertex number 'i', and decreases linearly to 0 towards the "triangle fan"
    # boundaries.
    hat_i = np.zeros(img.shape[:2])
    mask_i = np.zeros(img.shape[:2], dtype=bool)

    total_neighbor_area = 0.0 # Called "µ_i" in the paper.

    for tri_idx in neighs:
        tri = tris[tri_idx]
        assert i in tri
        v0, v1, v2 = np.take(uvs, tri, axis=0)
        total_neighbor_area += tri_areas[tri_idx]

        local_idx = tri.index(i)
        if backfacing[tri_idx]:
            mask, weights = rasterize(hat_i.shape, v0, v1, v2)
        else:
            mask, weights = rasterize(hat_i.shape, v0, v2, v1)
        hat_i[mask] = weights[..., local_idx][mask]
        mask_i[mask] = True
    
    num_samples = np.sum(mask_i)

    # Neighbor triangle area in pixels and the number of per-pixel samples considered in the weighted
    # average are very close but not exactly the same due to difference between analytical and rasterized areas.
    # I'm still computing the ratio here but you could possibly assume it to be unity and simplify this code.

    # Very small triangles may not rasterize even a single pixel.
    if num_samples > 0:
        b[i] = (total_neighbor_area / num_samples) * np.sum(hat_i[...,None] * img, axis=(0,1))

    if False:
        print('num_samples:', num_samples)
        print('total_neighbor_area:', total_neighbor_area)
        fig,ax=plt.subplots(3, figsize=(6,12))
        ax[0].imshow(hat_i)
        ax[1].imshow(mask_i)
        ax[2].imshow(hat_i[...,None] * img)
        plt.tight_layout()
        plt.show()

print(f"Build took: {time.time() - build_start:.3} s")

print("Solving")

import scipy.sparse

alpha = args.smoothing/100.0
R_scale = np.percentile(tri_areas, 50) # HACK: scale R matrix by triangle area since it's unitless (?)
A_reg = A + alpha * R_scale * R

x = np.zeros((N, 3))

solver_start = time.time()
for channel in tqdm(range(3)):
    x[..., channel], errorcode = scipy.sparse.linalg.cg(A_reg, b[..., channel])
    num_out_bounds = np.sum(np.logical_or(x < 0, x > 1))
    print(f"Number of vertices out of bounds: {num_out_bounds} = {(num_out_bounds/N)*100:.2f} %")
print(f"Solver took: {time.time() - solver_start:.3} s")

# Problem: Conjugate Gradient solver doesn't respect bounds so we have to clip the result.
x = np.clip(x, 0, 1)

if deduplicate:
    x_copy = x.copy()
    for old in range(x.shape[0]):
        new = old_vertex_id_to_new[old]
        x[old] = x_copy[new]

print(f"Saving GLB model {filename}")

color_rgb = (x*255).astype(np.uint8)
gltf = model_loader.add_vertex_colors(gltf, color_rgb, filename)
gltf.convert_buffers(pygltflib.BufferFormat.BINARYBLOB)
gltf.save_binary(args.output)
gltf.save("baked_plaintext.gltf")
model_loader.save_big_endian_dump(str(Path(args.output).with_suffix('.binm')), positions, raw_tris, color_rgb)
print("Saving done")

if args.show:
    print("Rasterizing the result for preview")

    img_result = np.zeros_like(img)
    draw_triangles(uvs, x, raw_tris, backfacing_raw, img_result)

    plot_shape = (1,2)
    figsize=(12,6)
    if img.shape[0] > img.shape[1]:
        plot_shape = (2,1)
        figsize=(figsize[1], figsize[0])
    fig,ax=plt.subplots(*plot_shape, figsize=figsize)
    ax.flatten()[0].imshow(img, vmin=0, vmax=1)
    ax.flatten()[0].set_title("Input image lightmap")
    ax.flatten()[1].imshow(img_result, vmin=0, vmax=1)
    ax.flatten()[1].set_title("Vertex color lightmap")
    plt.suptitle("Result")
    plt.tight_layout()
    plt.show()