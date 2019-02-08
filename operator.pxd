
from .graph.defs cimport gint
from .graph.defs import gint_dtype

cdef class BlockSparseMatrix:

    cdef public gint[:, :] block_offsets, block_shapes
    cdef public gint[:] data_offsets
    cdef public complex[:] data
    cdef complex* get(self, int block_idx)



cdef class _LocalOperator:

    cdef public gint check_hermiticity, sum
    cdef public object syst, onsite, _onsite_params_info
    cdef public gint[:, :]  where, _site_ranges
    cdef public BlockSparseMatrix _bound_onsite, _bound_hamiltonian

    cdef BlockSparseMatrix _eval_onsites(self, args, params)

    cdef BlockSparseMatrix _eval_hamiltonian(self, args, params)



cdef int _check_ham(complex[:, :] H, ham, args, params,
                    gint a, gint a_norbs, gint b, gint b_norbs,
                    int check_hermiticity)
