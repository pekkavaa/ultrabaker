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

# sRGB conversion functions by https://github.com/PetterS/opencv_srgb_gamma/blob/master/srgb.py

def to_linear(srgb):
	linear = np.float32(srgb)
	less = linear <= 0.04045
	linear[less] = linear[less] / 12.92
	linear[~less] = np.power((linear[~less] + 0.055) / 1.055, 2.4)
	return linear

    
def from_linear(linear):
	srgb = linear.copy()
	less = linear <= 0.0031308
	srgb[less] = linear[less] * 12.92
	srgb[~less] = 1.055 * np.power(linear[~less], 1.0 / 2.4) - 0.055
	return srgb

from ycocg import RGB_to_YCoCg, YCoCg_to_RGB

filename = args.input
print(f"Loading {filename}")
gltf = GLTF2().load(filename)
raw_positions, raw_normals, raw_uvs, raw_tris, img = model_loader.extract_pos_uvs_tris_img(gltf, filename)

# HACK: do processing in sRGB space
# img[...,:3] = to_linear(img[...,:3])
# img[...,:3] = RGB_to_YCoCg(img[...,:3])

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
    v0, v1, v2 = np.take(raw_uvs, tri, axis=0)
    area = cross2d(v1 - v0, v2 - v0) / 2
    backfacing_raw.append(area < 0)

deduplicate = True

if deduplicate:
    pos_to_id: dict[tuple, int] = dict()
    new_tris = []
    new_vertex_id_to_old = dict()
    old_vertex_id_to_new = dict()

    def get_location_key(p, n):
        x,y,z = p
        nx, ny, nz = n
        return (float(x), float(y), float(z), float(nx), float(ny), float(nz))


    for tri_idx, (a, b, c) in enumerate(raw_tris):
        new_tri = []
        for i0 in [a,b,c]:
            key = get_location_key(raw_positions[i0], raw_normals[i0])
            
            if key in pos_to_id:
                i1 = pos_to_id[key]
            else:
                i1 = i0
                pos_to_id[key] = i1
            
            new_vertex_id_to_old[i1] = i0
            old_vertex_id_to_new[i0] = i1
            new_tri.append(i1)
        
        new_tris.append(tuple(new_tri))
            

    for tri in new_tris:
        for i0 in tri:
            assert i0 in new_vertex_id_to_old

    tris = raw_tris
    positions = raw_positions
    normals = raw_normals
    uvs = raw_uvs
else:
    tris = raw_tris
    positions = raw_positions
    normals = raw_normals
    uvs = raw_uvs

N = len(uvs)


print('Finding vertex neighbors')

edge_tris = {}
vertex_tris = [[] for i in range(N)]
vertex_neighbors = [set() for i in range(N)]

# Maps a position-normal key to a set of oriented edges, vertex index pairs that is
edge_twins: dict[tuple, set[tuple]] = {}
# Vertices that overlap
vertex_twins: dict[tuple, set[tuple]] = {}

def get_edge_key(i,j):
    key_i = get_location_key(positions[i], normals[i])
    key_j = get_location_key(positions[j], normals[j])
    return (key_i, key_j)

for tri_idx, (a, b, c) in enumerate(tris):
    for i in [a,b,c]:
        vertex_twins.setdefault(get_location_key(positions[i], normals[i]), set()).add(i)

    for i, j in [(a,b), (b,c), (c,a)]:

        key_ij = get_edge_key(i,j)
        if key_ij not in edge_twins:
            edge_twins[key_ij] = set()
        edge_twins[key_ij].add((i,j))

        key_ji = get_edge_key(j,i)
        if key_ji not in edge_twins:
            edge_twins[key_ji] = set()
        edge_twins[key_ji].add((j,i))

for twins in edge_twins.values():
    head, *rest = list(twins)
    i,j = head
    if rest:
        for (k,l) in rest:
            assert np.all(positions[i] == positions[k])
            assert np.all(positions[j] == positions[l])
            assert np.all(normals[i] == normals[k])
            assert np.all(normals[j] == normals[l])

for vtwins in vertex_twins.values():
    i, *rest = list(twins)
    if rest:
        for j in rest:
            assert np.all(positions[i] == positions[j])
            assert np.all(normals[i] == normals[j])


for tri_idx, (a, b, c) in enumerate(tris):
    for i in [a,b,c]:
        twins = vertex_twins[get_location_key(positions[i], normals[i])]
        for ti in twins:
            vertex_tris[ti].append(tri_idx)

    # The 'edge_tris' array should contain *all* triangles incident to an edge used as a key
    # The 'vertex_neighbors' array has all vertex neighbor indices.
    # The 'vertex_tris' array has all the triangles the vertex is part of.
    for i, j in [(a,b), (b,c), (c,a)]:
        twins = edge_twins[get_edge_key(i,j)]
        assert (i,j) in twins

        vertex_neighbors[i].add(j)
        vertex_neighbors[j].add(i)

        for ti, tj in twins:
            edge_tris.setdefault((ti,tj), []).append(tri_idx)
            edge_tris.setdefault((tj,ti), []).append(tri_idx)

            vertex_neighbors[i].add(tj)
            vertex_neighbors[j].add(ti)
        
            # vertex_tris[ti].append(tri_idx)
            # vertex_tris[tj].append(tri_idx)


# I belive the above loop doesn't guarantee uniqueness of per-edge and per-vertex
# triangle lists so they are cleaned up here.

# Deduplicate per-edge triangle lists
for edge_key, tri_inds in edge_tris.items():
    edge_tris[edge_key] = list(set(tri_inds))

# Deduplicate per-vertex triangle list
for vidx, tri_inds in enumerate(vertex_tris):
    vertex_tris[vidx] = list(set(tri_inds))

# Validation. Check that each triangle has at least itself in per-edge triangle lists
for tri_idx, (a,b,c) in enumerate(tris):
    for i, j in [(a,b), (b,c), (c,a)]:
        neigh_tris = edge_tris[(i,j)]
        assert tri_idx in neigh_tris
    
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


cache_path = None
if args.input == "/home/user/dev/n64/hipoly_demo/work/lightmaps/bake_scene_trimmed.gltf":
    cache_path = "b_cache.npy"

import os

def get_location_key_ind(i):
    return get_location_key(positions[i], normals[i])


# check if file exists
if cache_path and os.path.exists(cache_path):
    print("Loading ", cache_path)
    b = np.load(cache_path)
else:
    inds = set()
    for a,b,c in new_tris:
        inds.add(a)
        inds.add(b)
        inds.add(c)
    inds = sorted(list(inds))

    print('Building the target vector b')
    b = np.zeros((N,3))

    # for i in tqdm(range(N)):
    for i in tqdm(inds):
        # Find triangles that neighbor this triangle.
        neigh_tris = vertex_tris[i]

        # Sample a linear "hat" function in the pixel grid that is 1 directly on the
        # the vertex number 'i', and decreases linearly to 0 towards the "triangle fan"
        # boundaries.
        hat_i = np.zeros(img.shape[:2])
        mask_i = np.zeros(img.shape[:2], dtype=bool)

        total_neighbor_area = 0.0 # Called "µ_i" in the paper.

        for tri_idx in neigh_tris:
            tri = tris[tri_idx]
            v0, v1, v2 = np.take(uvs, tri, axis=0)
            total_neighbor_area += tri_areas[tri_idx]

            # HACK: We need a local index for the current vertex in a given triangle.
            #       We use the position-normal key for it because we can't use a simple
            #       search through triangle's indices. A triangle can be an UV-space disjoint
            #       neighbor that has its own vertex indices even though they are neighbors
            #       in world space.
            local_idx = None
            key_i = get_location_key_ind(i)
            for local_j, j in enumerate(tri):
                if get_location_key_ind(j) == key_i:
                    local_idx = local_j
                    break
            assert local_idx is not None

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
            sample_grid = hat_i[...,None] * img
            b[i] = (total_neighbor_area / num_samples) * np.sum(sample_grid, axis=(0,1))

        if num_samples > 0 and False:
            print('num_samples:', num_samples)
            print('total_neighbor_area:', total_neighbor_area)
            print("neigh_tris:", neigh_tris)
            fig,axes=plt.subplots(2,2, figsize=(12,12))
            ax=axes.flatten()
            fig.suptitle(f"{i=} with {len(neigh_tris)} triangle neighbors")
            ax[0].imshow(hat_i)
            ax[0].set_title("hat_i")
            ax[1].imshow(mask_i)
            ax[1].set_title("mask_i")
            ax[2].imshow(sample_grid)
            ax[2].set_title("sample_grid")
            

            plt.tight_layout()
            plt.show()

    print(f"Build took: {time.time() - build_start:.3} s")
    if cache_path:
        np.save(cache_path, b)

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


# YCoCg results were identical to RGB so it's disabled.
# x = YCoCg_to_RGB(x)
# Conjugate Gradient solver doesn't respect bounds so we have to clip the result.
x = np.clip(x, 0, 1)
# The GLTF format expects vertex colors to be in linear space.
# Therefore we don't do gamma-to-linear conversion here.
# x = from_linear(x)

# HACK: sRGB processing
x = to_linear(x)


if deduplicate:
    x_copy = x.copy()
    for old, new in old_vertex_id_to_new.items():
        x[old] = x_copy[new]

print(f"Saving GLB model {filename}")

color_rgb = (x*255).astype(np.uint8)
gltf = model_loader.add_vertex_colors(gltf, color_rgb, filename)
gltf.save_binary(args.output)
gltf.save("baked_plaintext.gltf")
model_loader.save_big_endian_dump(str(Path(args.output).with_suffix('.binm')), positions, raw_tris, color_rgb)
print("Saving done")

if args.show:
    print("Rasterizing the result for preview")

    img_result = np.zeros_like(img)
    draw_triangles(uvs, x, tris, backfacing_raw, img_result)

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