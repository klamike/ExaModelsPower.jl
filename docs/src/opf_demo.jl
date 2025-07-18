# # [Static OPF](@id opf_demo)

# ExaModelsPower.jl can model large-scale optimal power flow (OPF) problems using the ExaModels package to generate models that can be solved using either CPU or GPU. This tutorial will demonstrate how ExaModelsPower.jl can be leveraged to solve different versions of the OPF, and how the user can customize the solving technique to better match their needs. Currently, all models generated by ExaModelsPower represent the full, AC version of the OPF formulation without any simplifications. 

# The latest version of ExaModelsPower can be installed in julia as so. Additionally, in order to develop models that can be solved on the GPU, CUDA is required. 
using ExaModelsPower, CUDA

# In order to solve the ExaModels developed by ExaModelsPower, an NLP solver is required. ExaModels is compatible with MadNLP and Ipopt, but this tutorial will focus on MadNLP to demonstrate GPU solving capabilities.
using MadNLP, MadNLPGPU #, NLPModelsIpopt 

# Finally, we install ExaModels to allow solved models to be unpacked.
using ExaModels

# We will begin by constructing and solving a static OPF using the function opf_model. For the static OPF, the only input required is the filename for the OPF matpower file. The file does not need to be locally installed, and it will be automatically downloaded from __[power-grid-library](https://github.com/power-grid-lib/pglib-opf)__ if the file is not found in the user's data folder. If keywords are not specified, the numerical type will default to Float64, the backend will default to nothing (used on CPU) and the form will default to polar coordinates. 
model, vars, cons = opf_model(
    "pglib_opf_case118_ieee.m";
    backend = CUDABackend(),
    form = :polar,
    T = Float64
);
model

# Once the model is built, we can generate a solution using MadNLP.
result = madnlp(model; tol=1e-6)

# Once a solution has been generated, the values of any of the variables in the model can be unpacked using the vars NamedTuple.
solution(result, vars.vm)[1:10]

# Result also stores the objective value.
result.objective

# ExaModelsPower supports solving the OPF in either polar or rectangular coordinates.
model, vars, cons = opf_model(
    "pglib_opf_case118_ieee.m";
    form = :rect
)
result = madnlp(model; tol=1e-6)
result.objective

# In this case, the objective value and performance speed is comparable. However, for some cases, MadNLP can only solve the problem on one of the two available coordinate systems.
