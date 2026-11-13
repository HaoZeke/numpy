/*
 * This file implements assignment from an ndarray to another ndarray.
 *
 * Written by Mark Wiebe (mwwiebe@gmail.com)
 * Copyright (c) 2011 by Enthought, Inc.
 *
 * See LICENSE.txt for the license.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "npy_config.h"


#include "convert_datatype.h"
#include "methods.h"
#include "shape.h"
#include "lowlevel_strided_loops.h"

#include "array_assign.h"
#include "dtype_transfer.h"

#include "umathmodule.h"

#include <stdint.h>
#include <string.h>

#define NPY_ALIGNED_CASTING_FLAG 1

/*
 * Check that array data is both uint-aligned and true-aligned for all array
 * elements, as required by the copy/casting code in lowlevel_strided_loops.c
 */
NPY_NO_EXPORT int
copycast_isaligned(int ndim, npy_intp const *shape,
        PyArray_Descr *dtype, char *data, npy_intp const *strides)
{
    int aligned;
    int big_aln, small_aln;

    int uint_aln = npy_uint_alignment(dtype->elsize);
    int true_aln = dtype->alignment;

    /* uint alignment can be 0, meaning not uint alignable */
    if (uint_aln == 0) {
        return 0;
    }

    /*
     * As an optimization, it is unnecessary to check the alignment to the
     * smaller of (uint_aln, true_aln) if the data is aligned to the bigger of
     * the two and the big is a multiple of the small aln. We check the bigger
     * one first and only check the smaller if necessary.
     */
    if (true_aln >= uint_aln) {
        big_aln = true_aln;
        small_aln = uint_aln;
    }
    else {
        big_aln = uint_aln;
        small_aln = true_aln;
    }

    aligned = raw_array_is_aligned(ndim, shape, data, strides, big_aln);
    if (aligned && big_aln % small_aln != 0) {
        aligned = raw_array_is_aligned(ndim, shape, data, strides, small_aln);
    }
    return aligned;
}

/*
 * A C-ordered matrix copied into Fortran order does not coalesce.
 * PrepareTwoRawArrayIter sorts by the destination stride, so the inner
 * loop writes contiguous items while reading with the other axis's
 * stride (a full row apart for float64). That is about 2 GB/s.
 * Tiles of 16 (8 for rank 3) keep a block of both layouts in cache.
 *
 * Only plain numeric items of 1, 2, 4, 8 or 16 bytes, already aligned,
 * with no cast. Anything else, including overlap that was not copied
 * aside by the caller, objects, and arrays that still fit in cache,
 * stays on the strided transfer loop. Returns 1 if this function did
 * the copy, 0 if the caller should use the normal path.
 */
#define NPY_TILE_COPY2(TYPE) \
    do { \
        for (ii = 0; ii < n0; ii += tile) { \
            ihi = ii + tile < n0 ? ii + tile : n0; \
            for (jj = 0; jj < n1; jj += tile) { \
                jhi = jj + tile < n1 ? jj + tile : n1; \
                for (i = ii; i < ihi; ++i) { \
                    for (j = jj; j < jhi; ++j) { \
                        *(TYPE *)(dp + i * ds0 + j * ds1) = \
                                *(const TYPE *)(sp + i * ss0 + j * ss1); \
                    } \
                } \
            } \
        } \
    } while (0)

#define NPY_TILE_COPY3(TYPE) \
    do { \
        for (ii = 0; ii < n0; ii += tile3) { \
            ihi = ii + tile3 < n0 ? ii + tile3 : n0; \
            for (jj = 0; jj < n1; jj += tile3) { \
                jhi = jj + tile3 < n1 ? jj + tile3 : n1; \
                for (kk = 0; kk < n2; kk += tile3) { \
                    khi = kk + tile3 < n2 ? kk + tile3 : n2; \
                    for (i = ii; i < ihi; ++i) { \
                        for (j = jj; j < jhi; ++j) { \
                            for (k = kk; k < khi; ++k) { \
                                *(TYPE *)(dp + i * ds0 + j * ds1 + k * ds2) = \
                                        *(const TYPE *)(sp + i * ss0 + \
                                                        j * ss1 + k * ss2); \
                            } \
                        } \
                    } \
                } \
            } \
        } \
    } while (0)

static int
raw_array_tile_transpose(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp const *src_strides,
        int aligned)
{
    int itemsize, kind;
    npy_intp nitems, n0, n1, n2;
    npy_intp ds0, ds1, ds2, ss0, ss1, ss2;
    npy_intp ii, jj, kk, i, j, k, ihi, jhi, khi;
    const npy_intp tile = 16;
    const npy_intp tile3 = 8;
    char *dp;
    const char *sp;
    int out_ndim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_it[NPY_MAXDIMS], src_it[NPY_MAXDIMS];
    char *dst_tmp, *src_tmp;
    NPY_BEGIN_THREADS_DEF;

    if ((ndim != 2 && ndim != 3) || !aligned) {
        return 0;
    }
    if (dst_dtype->type_num != src_dtype->type_num ||
            dst_dtype->elsize != src_dtype->elsize) {
        return 0;
    }
    kind = dst_dtype->type_num;
    if (!(kind == NPY_BOOL || kind == NPY_BYTE || kind == NPY_UBYTE ||
            kind == NPY_SHORT || kind == NPY_USHORT ||
            kind == NPY_INT || kind == NPY_UINT ||
            kind == NPY_LONG || kind == NPY_ULONG ||
            kind == NPY_LONGLONG || kind == NPY_ULONGLONG ||
            kind == NPY_FLOAT || kind == NPY_DOUBLE || kind == NPY_LONGDOUBLE ||
            kind == NPY_CFLOAT || kind == NPY_CDOUBLE ||
            kind == NPY_CLONGDOUBLE)) {
        return 0;
    }
    itemsize = dst_dtype->elsize;
    if (itemsize != 1 && itemsize != 2 && itemsize != 4 &&
            itemsize != 8 && itemsize != 16) {
        return 0;
    }
    nitems = 1;
    for (i = 0; i < ndim; ++i) {
        if (shape[i] <= 0 ||
                (dst_strides[i] % itemsize) != 0 ||
                (src_strides[i] % itemsize) != 0) {
            return 0;
        }
        nitems *= shape[i];
    }
    /* NumPy's inner loop is faster while the transpose fits in cache. */
    if (nitems <= ((npy_intp)8 << 20) / itemsize) {
        return 0;
    }
    if (((uintptr_t)dst_data % (uintptr_t)itemsize) != 0 ||
            ((uintptr_t)src_data % (uintptr_t)itemsize) != 0) {
        return 0;
    }
    /*
     * Same-order copies coalesce to one contiguous or unit-stride
     * transfer. That path is already a memcpy. Only the case that
     * stays two- or three-dimensional is a transpose.
     */
    dst_tmp = dst_data;
    src_tmp = src_data;
    if (PyArray_PrepareTwoRawArrayIter(
                ndim, shape, dst_tmp, dst_strides, src_tmp, src_strides,
                &out_ndim, shape_it, &dst_tmp, dst_it, &src_tmp, src_it) < 0 ||
            out_ndim < 2) {
        return 0;
    }

    n0 = shape[0];
    n1 = shape[1];
    ds0 = dst_strides[0];
    ds1 = dst_strides[1];
    ss0 = src_strides[0];
    ss1 = src_strides[1];
    dp = dst_data;
    sp = src_data;

    NPY_BEGIN_THREADS;
    if (ndim == 2) {
        switch (itemsize) {
        case 1: NPY_TILE_COPY2(npy_uint8); break;
        case 2: NPY_TILE_COPY2(npy_uint16); break;
        case 4: NPY_TILE_COPY2(npy_uint32); break;
        case 8: NPY_TILE_COPY2(npy_uint64); break;
        default:
            for (ii = 0; ii < n0; ii += tile) {
                ihi = ii + tile < n0 ? ii + tile : n0;
                for (jj = 0; jj < n1; jj += tile) {
                    jhi = jj + tile < n1 ? jj + tile : n1;
                    for (i = ii; i < ihi; ++i) {
                        for (j = jj; j < jhi; ++j) {
                            memcpy(dp + i * ds0 + j * ds1,
                                   sp + i * ss0 + j * ss1, 16);
                        }
                    }
                }
            }
            break;
        }
    }
    else {
        n2 = shape[2];
        ds2 = dst_strides[2];
        ss2 = src_strides[2];
        switch (itemsize) {
        case 1: NPY_TILE_COPY3(npy_uint8); break;
        case 2: NPY_TILE_COPY3(npy_uint16); break;
        case 4: NPY_TILE_COPY3(npy_uint32); break;
        case 8: NPY_TILE_COPY3(npy_uint64); break;
        default:
            for (ii = 0; ii < n0; ii += tile3) {
                ihi = ii + tile3 < n0 ? ii + tile3 : n0;
                for (jj = 0; jj < n1; jj += tile3) {
                    jhi = jj + tile3 < n1 ? jj + tile3 : n1;
                    for (kk = 0; kk < n2; kk += tile3) {
                        khi = kk + tile3 < n2 ? kk + tile3 : n2;
                        for (i = ii; i < ihi; ++i) {
                            for (j = jj; j < jhi; ++j) {
                                for (k = kk; k < khi; ++k) {
                                    memcpy(dp + i * ds0 + j * ds1 + k * ds2,
                                           sp + i * ss0 + j * ss1 + k * ss2,
                                           16);
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
    }
    NPY_END_THREADS;
    return 1;
}

#undef NPY_TILE_COPY2
#undef NPY_TILE_COPY3

/*
 * Assigns the array from 'src' to 'dst'. The strides must already have
 * been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_array(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp const *src_strides,
        int flags)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned = (flags & NPY_ALIGNED_CASTING_FLAG) != 0;
    int same_value_cast = (flags & NPY_SAME_VALUE_CASTING_FLAG) != 0;

    NPY_BEGIN_THREADS_DEF;

    /*
     * A transposed rank-2 or rank-3 copy does not coalesce, and the
     * stride sort below makes the inner loop jump across the source.
     * Tile that case. Overlap of ndim > 1 is copied aside by the caller.
     */
    if (raw_array_tile_transpose(
                ndim, shape,
                dst_dtype, dst_data, dst_strides,
                src_dtype, src_data, src_strides, aligned)) {
        return 0;
    }

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareTwoRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays and
     * opposite strides cause a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS method_flags;
    if (PyArray_GetDTypeTransferFunction(aligned,
                        src_strides_it[0], dst_strides_it[0],
                        src_dtype, dst_dtype,
                        0,
                        &cast_info, &method_flags) != NPY_SUCCEED) {
        return -1;
    }

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char*)&src_data);
    }

    if (same_value_cast) {
        cast_info.context.flags |= NPY_SAME_VALUE_CONTEXT_FLAG;
    }

    /* Ensure number of elements exceeds threshold for threading */
    if (!(method_flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    npy_intp strides[2] = {src_strides_it[0], dst_strides_it[0]};

    int result = 0;
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        /* Process the innermost dimension */
        char *args[2] = {src_data, dst_data};
        result = cast_info.func(&cast_info.context,
                                args, &shape_it[0], strides,
                                cast_info.auxdata);
        if (result < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it);

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier((char*)&src_data);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }

    return 0;
fail:
    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    return -1;
}

/*
 * Assigns the array from 'src' to 'dst, wherever the 'wheremask'
 * value is True. The strides must already have been broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_array(int ndim, npy_intp const *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp const *dst_strides,
        PyArray_Descr *src_dtype, char *src_data, npy_intp const *src_strides,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp const *wheremask_strides, int flags)
{
    int idim;
    npy_intp shape_it[NPY_MAXDIMS];
    npy_intp dst_strides_it[NPY_MAXDIMS];
    npy_intp src_strides_it[NPY_MAXDIMS];
    npy_intp wheremask_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    int aligned = (flags & NPY_ALIGNED_CASTING_FLAG) != 0;
    int same_value_cast = (flags & NPY_SAME_VALUE_CASTING_FLAG) != 0;

    NPY_BEGIN_THREADS_DEF;

    /* Use raw iteration with no heap allocation */
    if (PyArray_PrepareThreeRawArrayIter(
                    ndim, shape,
                    dst_data, dst_strides,
                    src_data, src_strides,
                    wheremask_data, wheremask_strides,
                    &ndim, shape_it,
                    &dst_data, dst_strides_it,
                    &src_data, src_strides_it,
                    &wheremask_data, wheremask_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays cause
     * a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
                src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        wheremask_data += (shape_it[0] - 1) * wheremask_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
        wheremask_strides_it[0] = -wheremask_strides_it[0];
    }

    /* Get the function to do the casting */
    NPY_cast_info cast_info;
    NPY_ARRAYMETHOD_FLAGS method_flags;
    if (PyArray_GetMaskedDTypeTransferFunction(aligned,
                        src_strides_it[0],
                        dst_strides_it[0],
                        wheremask_strides_it[0],
                        src_dtype, dst_dtype, wheremask_dtype,
                        0,
                        &cast_info, &method_flags) != NPY_SUCCEED) {
        return -1;
    }
    if (same_value_cast) {
        cast_info.context.flags |= NPY_SAME_VALUE_CONTEXT_FLAG;
    }

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier(src_data);
    }
    if (!(method_flags & NPY_METH_REQUIRES_PYAPI)) {
        npy_intp nitems = 1, i;
        for (i = 0; i < ndim; i++) {
            nitems *= shape_it[i];
        }
        NPY_BEGIN_THREADS_THRESHOLDED(nitems);
    }

    npy_intp strides[2] = {src_strides_it[0], dst_strides_it[0]};

    int result = 0;
    NPY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        PyArray_MaskedStridedUnaryOp *stransfer;
        stransfer = (PyArray_MaskedStridedUnaryOp *)cast_info.func;

        /* Process the innermost dimension */
        char *args[2] = {src_data, dst_data};
        result = stransfer(&cast_info.context,
                            args, &shape_it[0], strides,
                            (npy_bool *)wheremask_data, wheremask_strides_it[0],
                            cast_info.auxdata);
        if (result < 0) {
            goto fail;
        }
    } NPY_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it,
                            wheremask_data, wheremask_strides_it);

    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);

    if (!(method_flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        int fpes = npy_get_floatstatus_barrier(src_data);
        if (fpes && PyUFunc_GiveFloatingpointErrors("cast", fpes) < 0) {
            return -1;
        }
    }
    return 0;
fail:
    NPY_END_THREADS;
    NPY_cast_info_xfree(&cast_info);
    return -1;
}

/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array.
 * src: The source array.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_AssignArray(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting)
{
    int copied_src = 0;
    npy_intp src_strides[NPY_MAXDIMS];

    /* Use array_assign_scalar if 'src' NDIM is 0 */
    if (PyArray_NDIM(src) == 0) {
        return PyArray_AssignRawScalar(
                            dst, PyArray_DESCR(src), PyArray_DATA(src),
                            wheremask, casting);
    }

    /*
     * Performance fix for expressions like "a[1000:6000] += x".  In this
     * case, first an in-place add is done, followed by an assignment,
     * equivalently expressed like this:
     *
     *   tmp = a[1000:6000]   # Calls array_subscript in mapping.c
     *   np.add(tmp, x, tmp)
     *   a[1000:6000] = tmp   # Calls array_assign_subscript in mapping.c
     *
     * In the assignment the underlying data type, shape, strides, and
     * data pointers are identical, but src != dst because they are separately
     * generated slices.  By detecting this and skipping the redundant
     * copy of values to themselves, we potentially give a big speed boost.
     *
     * Note that we don't call EquivTypes, because usually the exact same
     * dtype object will appear, and we don't want to slow things down
     * with a complicated comparison.  The comparisons are ordered to
     * try and reject this with as little work as possible.
     */
    if (PyArray_DATA(src) == PyArray_DATA(dst) &&
                        PyArray_DESCR(src) == PyArray_DESCR(dst) &&
                        PyArray_NDIM(src) == PyArray_NDIM(dst) &&
                        PyArray_CompareLists(PyArray_DIMS(src),
                                             PyArray_DIMS(dst),
                                             PyArray_NDIM(src)) &&
                        PyArray_CompareLists(PyArray_STRIDES(src),
                                             PyArray_STRIDES(dst),
                                             PyArray_NDIM(src))) {
        /*printf("Redundant copy operation detected\n");*/
        return 0;
    }

    if (PyArray_FailUnlessWriteable(dst, "assignment destination") < 0) {
        goto fail;
    }

    /* Check the casting rule */
    if (!PyArray_CanCastTypeTo(PyArray_DESCR(src),
                                PyArray_DESCR(dst), casting)) {
        npy_set_invalid_cast_error(
                PyArray_DESCR(src), PyArray_DESCR(dst), casting, NPY_FALSE);
        goto fail;
    }

    /*
     * When ndim is 1 and the strides point in the same direction,
     * the lower-level inner loop handles copying
     * of overlapping data. For bigger ndim and opposite-strided 1D
     * data, we make a temporary copy of 'src' if 'src' and 'dst' overlap.'
     */
    if (((PyArray_NDIM(dst) == 1 && PyArray_NDIM(src) >= 1 &&
                    PyArray_STRIDES(dst)[0] *
                            PyArray_STRIDES(src)[PyArray_NDIM(src) - 1] < 0) ||
                    PyArray_NDIM(dst) > 1 || PyArray_HASFIELDS(dst)) &&
                    arrays_overlap(src, dst)) {
        PyArrayObject *tmp;

        /*
         * Allocate a temporary copy array.
         */
        tmp = (PyArrayObject *)PyArray_NewLikeArray(dst,
                                        NPY_KEEPORDER, NULL, 0);
        if (tmp == NULL) {
            goto fail;
        }

        if (PyArray_AssignArray(tmp, src, NULL, NPY_UNSAFE_CASTING) < 0) {
            Py_DECREF(tmp);
            goto fail;
        }

        src = tmp;
        copied_src = 1;
    }

    /* Broadcast 'src' to 'dst' for raw iteration */
    if (PyArray_NDIM(src) > PyArray_NDIM(dst)) {
        int ndim_tmp = PyArray_NDIM(src);
        npy_intp *src_shape_tmp = PyArray_DIMS(src);
        npy_intp *src_strides_tmp = PyArray_STRIDES(src);
        /*
         * As a special case for backwards compatibility, strip
         * away unit dimensions from the left of 'src'
         */
        while (ndim_tmp > PyArray_NDIM(dst) && src_shape_tmp[0] == 1) {
            --ndim_tmp;
            ++src_shape_tmp;
            ++src_strides_tmp;
        }

        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    ndim_tmp, src_shape_tmp,
                    src_strides_tmp, "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }
    else {
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(src), PyArray_DIMS(src),
                    PyArray_STRIDES(src), "input array",
                    src_strides) < 0) {
            goto fail;
        }
    }

    /* optimization: scalar boolean mask */
    if (wheremask != NULL &&
            PyArray_NDIM(wheremask) == 0 &&
            PyArray_DESCR(wheremask)->type_num == NPY_BOOL) {
        npy_bool value = *(npy_bool *)PyArray_DATA(wheremask);
        if (value) {
            /* where=True is the same as no where at all */
            wheremask = NULL;
        }
        else {
            /* where=False copies nothing */
            if (copied_src) {
                Py_DECREF(src);
            }
            return 0;
        }
    }

    int flags = (NPY_SAME_VALUE_CASTING_FLAG & casting);
    if (copycast_isaligned(PyArray_NDIM(dst), PyArray_DIMS(dst), PyArray_DESCR(dst),
                           PyArray_DATA(dst), PyArray_STRIDES(dst)) && 
        copycast_isaligned(PyArray_NDIM(dst), PyArray_DIMS(dst), PyArray_DESCR(src),
                           PyArray_DATA(src), src_strides)) {
        /* NPY_ALIGNED_CASTING_FLAG is internal to this file */
        flags |= NPY_ALIGNED_CASTING_FLAG;
    }

    if (wheremask == NULL) {
        /* A straightforward value assignment */
        /* Do the assignment with raw array iteration */
        if (raw_array_assign_array(PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                PyArray_DESCR(src), PyArray_DATA(src), src_strides, flags) < 0){
            goto fail;
        }
    }
    else {
        npy_intp wheremask_strides[NPY_MAXDIMS];

        /* Broadcast the wheremask to 'dst' for raw iteration */
        if (broadcast_strides(PyArray_NDIM(dst), PyArray_DIMS(dst),
                    PyArray_NDIM(wheremask), PyArray_DIMS(wheremask),
                    PyArray_STRIDES(wheremask), "where mask",
                    wheremask_strides) < 0) {
            goto fail;
        }

        /* A straightforward where-masked assignment */
         /* Do the masked assignment with raw array iteration */
        if (raw_array_wheremasked_assign_array(
                PyArray_NDIM(dst), PyArray_DIMS(dst),
                PyArray_DESCR(dst), PyArray_DATA(dst), PyArray_STRIDES(dst),
                PyArray_DESCR(src), PyArray_DATA(src), src_strides,
                PyArray_DESCR(wheremask), PyArray_DATA(wheremask),
                        wheremask_strides, flags) < 0) {
            goto fail;
        }
    }

    if (copied_src) {
        Py_DECREF(src);
    }
    return 0;

fail:
    if (copied_src) {
        Py_DECREF(src);
    }
    return -1;
}
