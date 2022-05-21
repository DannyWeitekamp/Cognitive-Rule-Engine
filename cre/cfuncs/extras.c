#include <string.h>
#include "numba/core/runtime/nrt_external.h"
#include "numba/core/runtime/nrt.h"

#include "extras.h" //if you don't have a header file you will get a warning


/* MemInfo is not exposed by nrt.h so we need to redefine it if we want to use it. */
struct MemInfo {
    size_t            refct;
    NRT_dtor_function dtor;
    void              *dtor_info;
    void              *data;
    size_t            size;    /* only used for NRT allocated memory */
    NRT_ExternalAllocator *external_allocator;
};

// Copies a meminfo and the data it points to.
// But does not incref object members. 
long NRT_MemInfo_copy_unsafe(void* nrt, long mi) {
    struct MemInfo* new_mi;
    struct MemInfo* old_mi;
    if(mi){
        old_mi = (struct MemInfo*) mi;
        new_mi = (struct MemInfo*) ((NRT_api_functions *) nrt)->allocate(old_mi->size);

        memcpy(new_mi->data, old_mi->data, old_mi->size);
        //Copy everything except refct and data
        new_mi->refct = 1;
        new_mi->dtor = old_mi->dtor;
        new_mi->dtor_info = old_mi->dtor_info;
        new_mi->external_allocator = old_mi->external_allocator;
        // return (NRT_MemInfo*) new_mi;
        return (long) new_mi;
        
    }else{
        return (long) NULL;
    }
}
