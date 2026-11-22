import pyomo.environ as pyo

# Create optimization model
model = pyo.ConcreteModel()

# Decision variables
model.x = pyo.Var(domain=pyo.NonNegativeReals)
model.y = pyo.Var(domain=pyo.NonNegativeReals)

# Objective function: Minimize cost
model.objective = pyo.Objective(
    expr=3 * model.x + 2 * model.y,
    sense=pyo.minimize
)

# Constraints
model.constraint1 = pyo.Constraint(
    expr=model.x + model.y >= 10
)

model.constraint2 = pyo.Constraint(
    expr=model.x >= 2
)

# Solve
solver = pyo.SolverFactory("appsi_highs")
result = solver.solve(model)

print("Solver status:", result.solver.status)
print("Termination:", result.solver.termination_condition)

print("Optimal x:", pyo.value(model.x))
print("Optimal y:", pyo.value(model.y))
print("Minimum cost:", pyo.value(model.objective))