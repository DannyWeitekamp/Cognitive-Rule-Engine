#include <stdio.h>
#include <Python.h>
#include "numba/core/runtime/nrt_external.h"
#include "numba/core/runtime/nrt.h"

/* MemInfo is not exposed by nrt.h so we need to redefine it if we want to use it. */
struct MemInfo {
    size_t            refct;
    NRT_dtor_function dtor;
    void              *dtor_info;
    void              *data;
    size_t            size;    /* only used for NRT allocated memory */
    NRT_ExternalAllocator *external_allocator;
};

// Copies a meminfo and the data it points to
NRT_MemInfo* meminfo_copy_unsafe(NRT_api_functions *nrt, NRT_MemInfo *mi) {
    struct MemInfo* new_mi;
    struct MemInfo* old_mi;
    if(mi){
        old_mi = (struct MemInfo*) mi;
        new_mi = (struct MemInfo*) nrt->allocate(mi->size);

        memcpy(new_mi->data, mi->data, mi->size);
        //Copy everything except refct and data
        new_mi->refct = 1;
        new_mi->dtor = old_mi->dtor;
        new_mi->dtor_info = old_mi->dtor_info;
        new_mi->external_allocator = old_mi->external_allocator;
        return (NRT_MemInfo*) new_mi;
        
    }else{
        return NULL;
    }
}

/*** START : ext_methods  ***/
static PyMethodDef ext_methods[] = {
#define declmethod(func) { #func , ( PyCFunction )func , METH_VARARGS , NULL }
#define declmethod_noargs(func) { #func , ( PyCFunction )func , METH_NOARGS, NULL }
    declmethod(meminfo_copy_unsafe),
    { NULL },
#undef declmethod
};
/*** END : ext_methods  ***/

/*** START : build_c_helpers_dict() ***/
static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr(value);           \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

#define declmethod(func) _declpointer(#func, &NRT_##func)
#define declmethod_internal(func) _declpointer(#func, &func)

declmethod_internal(meminfo_copy_unsafe);

#undef declmethod
#undef declmethod_internal
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}
/*** END : build_c_helpers_dict() ***/


// Module Definition struct
static struct PyModuleDef cre_cfuncs = {
    PyModuleDef_HEAD_INIT,
    "cre_cfuncs",
    "Test Module",
    -1,
    ext_methods
};

// Initializes module using above struct
PyMODINIT_FUNC PyInit_cre_cfuncs(void)
{
    PyObject *m = PyModule_Create(&cre_cfuncs);
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return m;
}
