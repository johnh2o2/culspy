%module culspy
%include typemaps.i

%{
extern void initialize_cuda(int);
extern void set_frequency_params(int, float *, float, float, int *, float *);
extern void eval_LS_periodogram(int, int, float,float, float *, float *, float *);
extern void bootstrap_LS_periodogram(int, int, float,float,  float *, float *, float *, int, int);
extern void batch_eval_LS_periodogram(int*, int, int, float, float, float *, float *, float *);
extern void stream_eval_LS_periodogram(int*, int, int, float, float, float *, float *, float *);
extern float *get_pinned_float_array(int n);
extern float get_pinned_val(float *x,int i);
extern void set_pinned_val(float *x, int i, float val);

%}

extern void initialize_cuda(int);

%apply int *OUTPUT { int *OUTPUT1 };
extern void set_frequency_params(int, float *, float, float, int *OUTPUT1, float *OUTPUT);
extern void eval_LS_periodogram(int, int, float,float,  float *, float *, float *);
extern void bootstrap_LS_periodogram(int, int, float,float,  float *, float *, float *, int, int);
extern void batch_eval_LS_periodogram(int*, int, int, float, float, float *, float *, float *);
extern void stream_eval_LS_periodogram(int*, int, int, float, float, float *, float *, float *);
extern float *get_pinned_float_array(int n);
extern float get_pinned_val(float *x,int i);
extern void set_pinned_val(float *x, int i, float val);

%inline %{



float *get_float_array(int n){
        float *x = (float *)malloc(n * sizeof(float));
        return x;
}
float get_val(float *x,int i){
        return x[i];
}
void set_val(float *x, int i, float val){
        x[i] = val;
}


int *get_int_array(int n){
        int *x = (int *)malloc(n * sizeof(int));
        return x;
}
int get_ival(int *x,int i){
        return x[i];
}
void set_ival(int *x, int i, int val){
        x[i] = val;
}

int get_block_size(){
    return BLOCK_SIZE;
}
%}

%pythoncode %{
BLOCK_SIZE = _culspy.get_block_size()


def _convert_to_c_pinned(arr):
    N = len(arr);
    carr = _culspy.get_pinned_float_array(N);
    for i, val in enumerate(arr):
        _culspy.set_pinned_val(carr, i, val)
    return carr

def _convert_to_c(arr):
    N = len(arr);
    carr = _culspy.get_float_array(N);
    for i, val in enumerate(arr):
        _culspy.set_val(carr, i, val)
    return carr

def _int_convert_to_c(arr):
    N = len(arr);
    carr = _culspy.get_int_array(N);
    for i, val in enumerate(arr):
        _culspy.set_ival(carr, i, int(val))
    return carr


def _convert_to_py(carr, N):
    return [ _culspy.get_val(carr, i) for i in range(N) ]

def _int_convert_to_py(carr, N):
    return [ _culspy.get_ival(carr, i) for i in range(N) ]

def _pinned_convert_to_py(carr, N):
    return [ _culspy.get_pinned_val(carr, i) for i in range(N) ]

def _insettings(arr, settings):
    return [ v in settings for v in arr ]

def correct_nf(Nf):
    if Nf % BLOCK_SIZE != 0:
        return (BLOCK_SIZE - Nf % BLOCK_SIZE) % BLOCK_SIZE
    return Nf


def LSP(t, x, f_over=1.0, f_high=1.0, minf=0.0, maxf=None, Nf=None ):
    Nt = len(t)
    ct = _convert_to_c(t)
    cx = _convert_to_c(x)

    if maxf is not None and Nf is not None:
        Nf = correct_nf(Nf)
        df = (maxf - minf)/Nf
    else:
        ct = _convert_to_c(t)
        Nf0, df0 = _culspy.set_frequency_params(Nt, ct, f_over, f_high)
        if not Nf is None:
            Nf = correct_nf(Nf)
            df = (Nf0 * df0)/Nf
        else:
            Nf = Nf0
            df = df0
    
    cpower = _culspy.get_float_array(Nf)
    _culspy.eval_LS_periodogram(Nt, Nf, df, minf, ct, cx, cpower)

    freqs = [ minf + df * i for i in range(Nf) ]
    power = _convert_to_py(cpower, Nf)
    
    
    return freqs, power

def LSPbootstrap(t, x, f_over=1.0, f_high=1.0, minf=0.0, 
                            maxf=None, Nf=None, Nbootstrap=100,
                            use_gpu_to_get_max = True ):
    Nt = len(t)
    ct = _convert_to_c(t)
    cx = _convert_to_c(x)

    if use_gpu_to_get_max: gpumax = 1
    else: gpumax = 0

    if maxf is not None and Nf is not None:
        Nf = correct_nf(Nf)
        df = (maxf - minf)/Nf
    else:
        ct = _convert_to_c(t)
        Nf0, df0 = _culspy.set_frequency_params(Nt, ct, f_over, f_high)
        if not Nf is None:
            Nf = correct_nf(Nf)
            df = (Nf0 * df0)/Nf
        else:
            Nf = Nf0
            df = df0
            
    cmax_heights = _culspy.get_float_array(Nbootstrap)
    _culspy.bootstrap_LS_periodogram(Nt, Nf, df, minf, ct, cx, 
                                            cmax_heights, Nbootstrap, gpumax)

    max_heights = _convert_to_py(cmax_heights, Nbootstrap)

    return max_heights

def get_memory_estimate(Nts, Nf):
    Nlc = len(Nts)
    memory_estimate = 4 * 2 * sum(Nts) + 4 * Nf * Nlc
    memory_estimate/= 1E9
    return memory_estimate

def divide_into_batches(Nts, Nf, max_memory):
    memory_estimate = get_memory_estimate(Nts, Nf)

    if memory_estimate < max_memory: return [ [ i for i in range(Nlc) ] ]

    else:
        batch = []
        batches = []
        lcno = 0
        while lcno < Nlc:
            if get_memory_estimate([ Nts[i] for i in batch ], Nf) > memory_estimate:
                ilast = batch.pop()
                batches.append([ i for i in batch ])
                batch = [ ilast ]
            else:
                batch.append(lcno)
                lcno += 1
        if len(batch) > 1: batches.append(batch)

        return batches

def LSPbatch(t, x, minf, maxf, Nf, max_memory=4. ):
    if Nf is None or maxf is None or minf is None:
        raise ValueError("Must give minf, maxf and Nf")

    df = (maxf - minf)/Nf

    Nts = [ len(T) for T in t ]
    mem = get_memory_estimate(Nts, Nf) 
    if mem > max_memory:
        raise MemoryError("Memory estimate (%.3f GB) exceeds max_memory = %.3f GB"%(mem, max_memory))

    flat_t = [ tval for T in t for tval in T  ]
    flat_x = [ xval for X in x for xval in X  ]

    Nlc = len(t)

    cflat_t = _convert_to_c(flat_t)
    cflat_x = _convert_to_c(flat_x)
    cflat_p = _culspy.get_float_array(Nf * Nlc)

    cNts = _int_convert_to_c(Nts)
  
    _culspy.batch_eval_LS_periodogram(cNts, Nlc, Nf, df, minf, cflat_t, cflat_x, cflat_p)

    freqs = [ minf + df * i for i in range(Nf) ]
    all_power = _convert_to_py(cflat_p, Nf * Nlc)
    
    powers = []
    for lcno in range(Nlc):
        offset = sum([ n for i, n in enumerate(Nts) if i < lcno ])
        powers.append([ all_power[i] for i in range(offset, offset + Nts[lcno]) ])
        


    return freqs, powers

def LSPstream(t, x, minf, maxf, Nf, max_memory=4. ):
    if Nf is None or maxf is None or minf is None:
        raise ValueError("Must give minf, maxf and Nf")

    df = (maxf - minf)/Nf

    Nts = [ len(T) for T in t ]
    mem = get_memory_estimate(Nts, Nf) 
    if mem > max_memory:
        raise MemoryError("Memory estimate (%.3f GB) exceeds max_memory = %.3f GB"%(mem, max_memory))

    flat_t = [ tval for T in t for tval in T  ]
    flat_x = [ xval for X in x for xval in X  ]

    Nlc = len(t)

    cflat_t = _convert_to_pinned_c(flat_t)
    cflat_x = _convert_to_pinned_c(flat_x)
    cflat_p = _culspy.get_pinned_float_array(Nf * Nlc)

    cNts = _int_convert_to_c(Nts)
  
    _culspy.stream_eval_LS_periodogram(cNts, Nlc, Nf, df, minf, cflat_t, cflat_x, cflat_p)

    freqs = [ minf + df * i for i in range(Nf) ]
    all_power = _pinned_convert_to_py(cflat_p, Nf * Nlc)
    
    powers = []
    for lcno in range(Nlc):
        offset = sum([ n for i, n in enumerate(Nts) if i < lcno ])
        powers.append([ all_power[i] for i in range(offset, offset + Nts[lcno]) ])
        


    return freqs, powers

%}
