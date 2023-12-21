# This script generates an LF program that uses the 'pipeline' pattern
# It can generate as many pipelines stages as specified
#
# The reason this script exists is that currently, LF does not have a way of
# connecting pipeline stages together in a scalable approach

import sys
if len(sys.argv) == 2:
    NWORKERS = int(sys.argv[1])
else:
    print("Provide number of workers!")
    exit(1)

def gen_compute_part_x_cfnc(x: int):
    return """
    static float compute_part_""" + str(x) + """(float f) {
        lf_sleep(MSEC(10));
        return f + 1;
    }
    """

def gen_preamble():
    preamble = """preamble {=
    #include <platform.h>

    #define MAX_ITERATIONS (1000)
    #define NSTAGES (""" + str(NWORKERS) + """)
    #define TIMESTAMP_SIZE ((1 + NSTAGES) * MAX_ITERATIONS)
    
    #define IDX_BASE(i)         ((1 + NSTAGES) * i + (    0))
    #define IDX_DATA(i, j)      ((1 + NSTAGES) * i + (j + 1))
    """

    for i in range(NWORKERS):
        preamble += gen_compute_part_x_cfnc(i)
    
    preamble += """

    static void print_timings(uint32_t iter, instant_t *timestamps)
    {
        char printout[256];
        int len = sprintf(printout, "Iteration %i: Base: %lli, ", iter, timestamps[(1 + NSTAGES) * iter]);
        for (int j = 0; j < NSTAGES; j++) {
            if (timestamps[(1 + NSTAGES) * iter + (j + 1)] != 0) {
                len += sprintf(printout + len - 1, "{%i}: %lli, ", j, timestamps[(1 + NSTAGES) * iter + (j + 1)]) - 1;
            } else {
                len += sprintf(printout + len - 1, "{%i}: NA, ", j) - 1;
            }
        }
        
        printf("%s\\n", printout);
    }
=}"""
    return preamble

def gen_reaction_startup():
    reaction  = "    reaction(startup) -> " + ("".join(f"stage{x}.timestamps" + (", " if x != NWORKERS-1 else " ") for x in range(NWORKERS)))
    reaction += "{=\n"
    reaction += "        fp_assert(NSTAGES == NUMBER_OF_WORKERS, \"Number of stages not equal to number of workers!\\n\");\n"
    reaction += "        self->timestamps = malloc(TIMESTAMP_SIZE * sizeof(instant_t));\n\n"
    
    reaction += "".join(f"        lf_set(stage{x}.timestamps, self->timestamps);\n" for x in range(NWORKERS))
    return reaction + "    =}"

def gen_reaction_t():
    reaction = "    reaction(t) -> stage0.x, " + "".join(f"stage{x}.niterations" + (", " if x != NWORKERS-1 else " ") for x in range(NWORKERS))
    reaction += "{=\n"
    reaction += """        self->timestamps[IDX_BASE(self->niterations)] = lf_time_physical_elapsed();
        lf_set(stage0.x, 10.0);\n\n"""
    
    reaction += "".join(f"        lf_set(stage{x}.niterations, self->niterations);\n" for x in range(NWORKERS))

    return reaction + "    =}"

def gen_pipeline_stages():
    return "".join(f"    stage{x} = new PipelineStage(stage = {x});\n" for x in range(NWORKERS))

def gen_pipeline_connections():
    return "".join(f"    stage{x}.y -> stage{x+1}.x after 10 msec\n" for x in range(NWORKERS-1))

def gen_reaction_stagelast_dot_y():
    return "    reaction(stage" + str(NWORKERS-1) + ".y) {=\n" + \
        "        self->processed = stage" + str(NWORKERS-1) + ".y->value;\n" + \
        """        if (++self->niterations == MAX_ITERATIONS) {
            printf("request stop\\n");
            lf_request_stop();
        }
    =}"""

def gen_reaction_shutdown():
    return """    reaction(shutdown) {=
        // Remove the parts of the pipeline that were empty in the beginning
        for (int j = 0; j < NSTAGES; j++) {
            for (int k = 1; k < (NSTAGES - j); k++) {
                self->timestamps[(1 + NSTAGES) * j + (j + k + 1)] = 0;
            }
        }

        for (int i = 0; i < self->niterations; i++) {
            print_timings(i, self->timestamps);
        }

        free(self->timestamps);
    =}"""


def gen_main_reactor():
    return """main reactor(start: time = 100 msec) {
    state processed: float
    state niterations: int = 0
    state timestamps: instant_t[]

    timer t(0, 10 msec)\n\n""" + \
    gen_reaction_startup() + "\n\n" + \
    gen_reaction_t() + "\n\n" + \
    gen_pipeline_stages() + "\n" + \
    gen_pipeline_connections() + "\n" + \
    gen_reaction_stagelast_dot_y() + "\n\n" + \
    gen_reaction_shutdown() + "\n}"

def gen_pipeline_stage_reactor():
    return """reactor PipelineStage(stage: int = 0) {
    input x: float
    input niterations: int
    input timestamps: instant_t[]
    output y: float
    reaction(x, niterations, timestamps) -> y {=
        instant_t time_elapsed = lf_time_physical_elapsed();
        timestamps->value[IDX_DATA(niterations->value, self->stage)] = time_elapsed;

        float result = 0;
        switch (self->stage) {\n""" + \
    "".join(f"            case {x}: result = compute_part_{x}(x->value); break;\n" for x in range(NWORKERS)) + \
    """
            default: break;
        }
        lf_set(y, result);
    =}
}"""

def gen_PipelineKF_lf():
    return """target C {
    threading: true,
    logging: error,
    build: "../scripts/build_flexpret_unix.sh"
}\n\n""" + gen_preamble() + "\n\n" + gen_main_reactor() + "\n\n" + gen_pipeline_stage_reactor() + "\n"

with open(f"src/Pipeline" + str(NWORKERS) + ".lf", 'w') as f:
    f.write(gen_PipelineKF_lf())
