function build_dcopf(data, user_callback; backend = nothing, T = Float64, kwargs...)
    core = ExaCore(T; backend = backend)

    va = variable(core, length(data.bus))
    pg = variable(core, length(data.gen); lvar = data.pmin, uvar = data.pmax)

    branch_rate_a = [br.rate_a for br in data.branch]
    pf = variable(
        core,
        length(data.branch);
        lvar = -branch_rate_a,
        uvar = branch_rate_a
    )

    o = objective(core, gen_cost(g, pg[g.i]) for g in data.gen)

    c_ref_angle = constraint(core, c_ref_angle_polar(va[i]) for i in data.ref_buses)
    
    c_ohms_law = constraint(
        core,
        c_ohms_law_dcopf(br, pf[br.i], va[br.f_bus], va[br.t_bus])
        for br in data.branch
    )

    c_phase_angle_diff = constraint(
        core,
        c_phase_angle_diff_polar(b, va[b.f_bus], va[b.t_bus]) for b in data.branch;
        lcon = data.angmin,
        ucon = data.angmax,
    )

    c_active_power_balance = constraint(
        core,
        c_active_power_balance_demand_dcopf(b) for b in data.bus
    )
    constraint!(core, c_active_power_balance, g.bus => -pg[g.i] for g in data.gen)
    constraint!(core, c_active_power_balance, br.f_bus => pf[br.i] for br in data.branch)
    constraint!(core, c_active_power_balance, br.t_bus => -pf[br.i] for br in data.branch)

    vars = (
        va = va,
        pg = pg,
        pf = pf,
    )

    cons = (
        c_ref_angle = c_ref_angle,
        c_ohms_law = c_ohms_law,
        c_phase_angle_diff = c_phase_angle_diff,
        c_active_power_balance = c_active_power_balance,
    )

    vars2, cons2 = user_callback(core, vars, cons)

    model = ExaModel(core; kwargs...)

    vars = (; vars..., vars2...)
    cons = (; cons..., cons2...)

    return model, vars, cons
end

function dcopf_model(
    filename;
    backend = nothing,
    T = Float64,
    user_callback = dummy_extension,
    kwargs...,
)
    data = parse_ac_power_data(filename)
    data = convert_data(data, backend)

    return build_dcopf(data, user_callback; backend = backend, T = T, kwargs...)
end
