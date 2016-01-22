%module culspy
%include typemaps.i

%{
extern void initialize_cuda(int);
extern void set_frequency_params(int, float *, float, float, int *, float *);
extern void eval_LS_periodogram(int, int, float,float, float *, float *, float *);
%}

extern void initialize_cuda(int);

%apply int *OUTPUT { int *OUTPUT1 };
extern void set_frequency_params(int, float *, float, float, int *OUTPUT1, float *OUTPUT);
extern void eval_LS_periodogram(int, int, float,float,  float *, float *, float *);

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

int get_block_size(){
    return BLOCK_SIZE;
}
%}

%pythoncode %{
BLOCK_SIZE = _culspy.get_block_size()

def _convert_to_c(arr):
    N = len(arr);
    carr = _culspy.get_float_array(N);
    for i, val in enumerate(arr):
        _culspy.set_val(carr, i, val)
    return carr

def _convert_to_py(carr, N):
    return [ _culspy.get_val(carr, i) for i in range(N) ]

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

%}
