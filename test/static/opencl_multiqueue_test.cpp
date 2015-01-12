/*
Written on Ubuntu 14.04, AMD Radeon R9 290, fglrx 14.09.

Building and running:
g++ opencl_multiqueue_generate.cpp -I ../../include/ -L ../../bin -lHalide -o opencl_multiqueue_generate
LD_LIBRARY_PATH=../../bin ./opencl_multiqueue_generate
g++ opencl_multiqueue_test.cpp opencl_multiqueue.o -I ../../apps/support/ -lpthread -lOpenCL -o opencl_multiqueue_test
HL_OCL_DEVICE_TYPE=gpu ./opencl_multiqueue_test
*/


#include "opencl_multiqueue.h"
#include "../../include/HalideRuntime.h"
#include "../performance/clock.h"
#include <static_image.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>


namespace Halide { namespace Runtime { namespace Internal {
extern int create_opencl_context(void *user_context, cl_context *ctx, cl_command_queue *q);
extern const char *get_opencl_error_name(cl_int err);
}}}

// Container for cl context and command queue.
// Allows sharing same context between multiple queues
class UserContext {
public:
    UserContext()
    : context(0)
    , queue(0)
    , device(0)
    {
    }

    // Explicitly initialize this UserContext.
    // If sharedContext is given, another queue will be created to the same OpenCL context,
    // and the same device.
    int initialize(const UserContext& sharedContext = UserContext());

    int sync() {
        assert(queue);
        return clFinish(queue);
    }

    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
};

volatile int lock = 0;

extern "C" {

int halide_acquire_cl_context(void *user_context, cl_context *ctx, cl_command_queue *q, bool *auto_finish_ptr)
{
    cl_int error = CL_SUCCESS;
    UserContext* context = (UserContext*)user_context;

    if (!context) {
        // free_dev_buffer passes null user_context and it does not actually use ctx or q.
        // ClContext::ClContext asserts however that ctx and q are not NULL.
        // TODO: Should a global context and queue be created for this purpose for nothing,
        // or free_dev_buffer changed?
        // Return bogus, for now.
        *ctx = (cl_context)1;
        *q = (cl_command_queue)1;
        return 0;
    }

    while (__sync_lock_test_and_set(&lock, 1)) { }

    if ((!context->context) || (!context->queue)) {

        if (!context->context) {
            //printf("@Mon create context\n");
            error = Halide::Runtime::Internal::create_opencl_context(user_context, &context->context, &context->queue);
        } else {
            //printf("@Mon create queue\n");
            assert(!context->queue);
            assert(context->device); // assume this UserContext has been initialized with a sharedContext

            context->queue = clCreateCommandQueue(context->context, context->device, 0, &error);
        }
    }

    if (error == CL_SUCCESS) {
        assert(context->context);
        assert(context->queue);

        *ctx = context->context;
        *q = context->queue;
        *auto_finish_ptr = false;
    } else {
        printf("Error in halide_acquire_cl_context: %s\n", Halide::Runtime::Internal::get_opencl_error_name(error));
    }

    __sync_lock_release(&lock);

    return error;
}

int halide_release_cl_context(void *user_context) {
    while (__sync_lock_test_and_set(&lock, 1)) { }
    // TODO: what needs to be done here?
    __sync_lock_release(&lock);
    return 0;
}

} // extern "C"

int UserContext::initialize(const UserContext& sharedContext)
{
    context = sharedContext.context;
    if (sharedContext.queue) {
        clGetCommandQueueInfo(sharedContext.queue, CL_QUEUE_DEVICE, sizeof device, &device, NULL);
        assert(device);
    }

    bool auto_finish;
    return halide_acquire_cl_context((void*)this, &context, &queue, &auto_finish);
}

void benchmark(Image<float> input, Image<float> output, const UserContext& rootContext, int queues, int repeats, bool memoryTraffic)
{
    assert(queues > 0);
    UserContext* contexts = new UserContext[queues];
    contexts[0] = rootContext;
    for (int i = 1; i < queues; ++i) {
        contexts[i].initialize(rootContext);
    }

    double start_time = current_time();
    for (int i = 0; i < repeats; ++i) {
        UserContext* user_context = &contexts[i % queues];

        if (memoryTraffic) {
            input.set_host_dirty();
            input.copy_to_dev(user_context);
        }

        for (int j = 0; j < 10; ++j) {
            opencl_multiqueue(user_context, input, output);
        }

        if (memoryTraffic) {
            output.copy_to_host(user_context);
        }
    }

    for (int i = 0; i < queues; ++i) {
        contexts[i].sync();
    }
    double end_time = current_time();

    printf("%d\t%g\n", queues, (end_time - start_time));

    delete[] contexts;
}

void benchmarkQueues(Image<float> input, Image<float> output, const UserContext& rootContext, int queuesMax, bool memoryTraffic)
{
    printf("Queues\tTime (ms)\n");
    int repeats = 256;
    for (int queues = 1; queues <= queuesMax; ++queues) {
        benchmark(input, output, rootContext, queues, repeats, memoryTraffic);
    }
}


int main(int argc, char **argv) {    
    const int dim = 1024;
    Image<float> input(dim, dim);
    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
          input(x, y) = 1;
        }
    }
    Image<float> output(dim, dim);

    // init one context as root that holds the context, kernels, buffers etc.
    UserContext rootContext;
    rootContext.initialize();

    // build kernel and do any other init
    opencl_multiqueue(&rootContext, input, output);

    printf("With memory transfers:\n");
    benchmarkQueues(input, output, rootContext, 8, true);

    printf("\nWithout memory transfers:\n");
    benchmarkQueues(input, output, rootContext, 8, false);

    printf("Done.\n");
    return 0;
}
