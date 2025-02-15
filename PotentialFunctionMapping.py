import jax
import jax.numpy as jnp
from jax.lax import scan
import matplotlib
matplotlib.use('Agg')  # If running headless or in a script
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial


# Formulas
# 
#
#  Potential: V(x,y) = x^4 + y^4 + x^3 - 2*x*y^2 + a*(x^2 + y^2) + theta1*x + theta2*y

def potential_V(x, y):
    a = 2.0
    theta1 = 5.0
    theta2 = -5.0
    return (x**4 + y**4
            + x**3
            - 2.0*x*y**2
            + a*(x**2 + y**2)
            + theta1*x
            + theta2*y)


# Calculating the drift of potential
@jax.jit
def drift_f(X):
    a = 2.0
    theta1 = 5.0
    theta2 = -5.0
    x, y = X
    dVdx = 4.0*x**3 + 3.0*x**2 - 2.0*y**2 + 2.0*a*x + theta1
    dVdy = 4.0*y**3 - 4.0*x*y   + 2.0*a*y + theta2
    return jnp.array([-dVdx, -dVdy])

#Potential Landscape Plot

def plot_potential_landscape(x_min=-2.0, x_max=2.0,
                             y_min=-2.0, y_max=2.0,
                             Nx=80, Ny=80,
                             filename="potential_landscape_heavy.png"):

    print(f"DEBUG: plot_potential_landscape => domain=({x_min},{x_max})x({y_min},{y_max}), Nx={Nx}, Ny={Ny}")
    x_lin = jnp.linspace(x_min, x_max, Nx)
    y_lin = jnp.linspace(y_min, y_max, Ny)
    XX, YY = jnp.meshgrid(x_lin, y_lin, indexing='xy')

    def v_xy(x_, y_):
        return potential_V(x_, y_)
    V_vals = jax.vmap(jax.vmap(v_xy, in_axes=(0,None)), in_axes=(None,0))(x_lin, y_lin)

    def neg_grad_V(x_, y_):
        a=2.0
        theta1=5.0
        theta2=-5.0
        dVdx = 4.0*x_**3 + 3.0*x_**2 - 2.0*y_**2 + 2.0*a*x_ + theta1
        dVdy = 4.0*y_**3 - 4.0*x_*y_ + 2.0*a*y_ + theta2
        return -dVdx, -dVdy

    flow_map = jax.vmap(jax.vmap(neg_grad_V, in_axes=(0,None)), in_axes=(None,0))(x_lin, y_lin)
    U_flow = flow_map[0]
    V_flow = flow_map[1]

    plt.figure(figsize=(7,6))
    cp = plt.contourf(XX, YY, V_vals, levels=50, cmap='viridis')
    plt.colorbar(cp, label="Potential V(x,y)")
    plt.quiver(XX, YY, U_flow, V_flow, color='white', alpha=0.6)
    plt.title("Potential Landscape (a=2,theta1=5,theta2=-5)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

# Langevein 

@jax.jit
def euler_maruyama_step(X, params):
    
    D, dt, key = params
    key, subkey = jax.random.split(key, 2)
    f_val = drift_f(X)
    noise = jax.random.normal(subkey, shape=(2,))
    X_next = X + dt*f_val + jnp.sqrt(2*D)*jnp.sqrt(dt)*noise
    return (X_next, (D, dt, key)), X_next

@partial(jax.jit, static_argnums=(1,))
def single_trajectory_final(X_init, n_steps: int, D, dt, key):
    params = (D, dt, key)
    def step_fn(carry, _):
        return euler_maruyama_step(carry[0], carry[1])
    carry0 = (X_init, params)
    (_, traj) = scan(step_fn, carry0, None, length=n_steps)
    return traj[-1]


def chunked_langevin(X0,
                     D=0.2,
                     dt=0.01,
                     n_steps=50_000,
                     n_trajectories=20_000,
                     chunk_size=1000,
                     seed=42,
                     hist_bins=100,
                     hist_range=(-2.0,2.0)):

    n_steps_int = int(n_steps)

    print(f"DEBUG: chunked_langevin => D={D}, dt={dt}, n_steps={n_steps_int}, "
          f"n_trajectories={n_trajectories}, chunk_size={chunk_size}")

    start_time = time.time()

    key = jax.random.PRNGKey(seed)
    n_chunks = (n_trajectories + chunk_size - 1)//chunk_size

    import numpy as np
    hist2d = np.zeros((hist_bins, hist_bins), dtype=np.float64)
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    total_count = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_trajectories)
        actual_size = chunk_end - chunk_start
        if actual_size <= 0:
            break

        subkeys = jax.random.split(key, actual_size)
        key = subkeys[-1]

        X_inits = jnp.tile(X0, (actual_size,1))
        final_positions = jax.vmap(
            lambda x0_, k_: single_trajectory_final(x0_, n_steps_int, D, dt, k_)
        )(X_inits, subkeys)

        final_positions_np = np.array(final_positions)

        x_vals_chunk = final_positions_np[:,0]
        y_vals_chunk = final_positions_np[:,1]
        chunk_hist, _, _ = np.histogram2d(x_vals_chunk, y_vals_chunk,
                                          bins=hist_bins,
                                          range=[[hist_range[0],hist_range[1]],
                                                 [hist_range[0],hist_range[1]]])
        hist2d += chunk_hist

        sum_x += x_vals_chunk.sum()
        sum_y += y_vals_chunk.sum()
        sum_x2 += (x_vals_chunk**2).sum()
        sum_y2 += (y_vals_chunk**2).sum()
        total_count += actual_size

        print(f"  -> Processed chunk {chunk_idx+1}/{n_chunks} with size={actual_size}")

    if hist2d.sum() > 0:
        hist2d /= hist2d.sum()

    if total_count > 0:
        mean_x = sum_x / total_count
        mean_y = sum_y / total_count
        var_x = (sum_x2 / total_count) - mean_x**2
        var_y = (sum_y2 / total_count) - mean_y**2
        std_x = np.sqrt(max(var_x,0.0))
        std_y = np.sqrt(max(var_y,0.0))
    else:
        mean_x, mean_y, std_x, std_y = 0.0,0.0,0.0,0.0

    end_time = time.time()
    print(f"DEBUG: chunked_langevin total time ~{(end_time - start_time):.2f}s, "
          f"Processed {n_trajectories} trajectories in {n_chunks} chunks.")
    print(f"DEBUG: final x mean={mean_x:.3f} std={std_x:.3f}, y mean={mean_y:.3f} std={std_y:.3f}")

    return hist2d, mean_x, std_x, mean_y, std_y


def update_p(P, dx, dy, dt, D, x_grid, y_grid, Nx, Ny):

    def f_xy(i, j):
        xx = x_grid[i]
        yy = y_grid[j]
        a=2.0
        theta1=5.0
        theta2=-5.0
        dVdx = 4.0*xx**3 + 3.0*xx**2 - 2.0*yy**2 + 2.0*a*xx + theta1
        dVdy = 4.0*yy**3 - 4.0*xx*yy + 2.0*a*yy + theta2
        return jnp.array([-dVdx, -dVdy])

    def shift_left(M):
        return jnp.concatenate([M[:,0:1], M[:,:-1]], axis=1)
    def shift_right(M):
        return jnp.concatenate([M[:,1:], M[:,-1:]], axis=1)
    def shift_up(M):
        return jnp.concatenate([M[0:1,:], M[:-1,:]], axis=0)
    def shift_down(M):
        return jnp.concatenate([M[1:,:], M[-1:,:]], axis=0)

    fx_array = []
    fy_array = []
    for i in range(Nx):
        row_vals = []
        for j in range(Ny):
            fval = f_xy(i, j)
            row_vals.append(fval)
        row_vals = jnp.stack(row_vals, axis=0)
        fx_array.append(row_vals)
    fx_array = jnp.stack(fx_array, axis=0)
    fx_array = jnp.swapaxes(fx_array, 0,1)

    fx = fx_array[:,:,0]
    fy = fx_array[:,:,1]

    flux_x = fx * P
    flux_y = fy * P
    flux_x = flux_x.at[:,0].set(0.0)
    flux_x = flux_x.at[:,-1].set(0.0)
    flux_y = flux_y.at[0,:].set(0.0)
    flux_y = flux_y.at[-1,:].set(0.0)

    flux_x_left = shift_left(flux_x)
    div_x = (flux_x - flux_x_left)/dx
    flux_y_up = shift_up(flux_y)
    div_y = (flux_y - flux_y_up)/dy

    div_fP = div_x + div_y

    P_left  = shift_left(P)
    P_right = shift_right(P)
    P_up    = shift_up(P)
    P_down  = shift_down(P)

    lap_x = (P_left - 2.0*P + P_right)/(dx*dx)
    lap_y = (P_up   - 2.0*P + P_down )/(dy*dy)
    lap_P = lap_x + lap_y

    dPdt = -div_fP + D*lap_P
    P_new = P + dt*dPdt
    P_new = jnp.clip(P_new, 0.0, None)
    s = jnp.sum(P_new)
    P_new = jnp.where(s>0, P_new/s, P_new)
    return P_new

def solve_fokker_planck_2d(D=0.2,
                           x_min=-2.0, x_max=2.0, Nx=50,  # reduce Nx from 100 to 50
                           y_min=-2.0, y_max=2.0, Ny=50,  # reduce Ny from 100 to 50
                           dt=1e-5,                      # increase dt from 1e-6
                           T=0.005):                     # reduce T from 0.01
    print(f"DEBUG: solve_fokker_planck_2d => domain=({x_min},{x_max})x({y_min},{y_max}), "
          f"Nx={Nx}, Ny={Ny}, dt={dt}, T={T}, D={D}")
    start_time = time.time()

    Nx = int(Nx)
    Ny = int(Ny)
    x_grid = jnp.linspace(x_min, x_max, Nx)
    y_grid = jnp.linspace(y_min, y_max, Ny)
    dx = (x_max - x_min)/(Nx - 1)
    dy = (y_max - y_min)/(Ny - 1)

    def init_cond(x, y):
        sigma0 = 0.2
        r2 = x**2 + y**2
        return jnp.exp(-r2/(2*sigma0**2))

    XX, YY = jnp.meshgrid(x_grid, y_grid, indexing='xy')
    P0 = jax.vmap(jax.vmap(init_cond))(XX, YY)
    P0 = P0 / jnp.sum(P0)

    n_steps = int(T/dt)
    conv_data_list = []
    P_current = P0

    for i in range(n_steps):
        P_next = update_p(P_current, dx, dy, dt, D, x_grid, y_grid, Nx, Ny)
        diff_val = float(jnp.sum(jnp.abs(P_next - P_current)))
        conv_data_list.append(diff_val)
        P_current = P_next

    P_final = P_current
    conv_data_final = jnp.array(conv_data_list, dtype=jnp.float32)

    end_time = time.time()
    print(f"DEBUG: PDE took ~{(end_time - start_time):.2f}s, "
          f"last step diff={conv_data_list[-1]:.4e}")

    return x_grid, y_grid, P_final, conv_data_final

def main():
    overall_start = time.time()

    # Potential
    plot_potential_landscape()

    # Langevin
    X0 = jnp.array([0.0, 0.0])
    D_langevin = 0.2
    dt_langevin = 0.01
    n_steps_langevin = 50_000
    n_trajectories = 20_000
    chunk_size = 2000
    print("\n=== CHUNKED LANGEVIN SIMULATION ===")
    hist2d, mean_x, std_x, mean_y, std_y = chunked_langevin(
        X0,
        D=D_langevin,
        dt=dt_langevin,
        n_steps=n_steps_langevin,
        n_trajectories=n_trajectories,
        chunk_size=chunk_size
    )

    print(f"DEBUG: final x mean={mean_x:.3f} std={std_x:.3f}, "
          f"y mean={mean_y:.3f} std={std_y:.3f}")

    # Plot final histogram
    plt.figure(figsize=(7,6))
    plt.imshow(np.log(hist2d.T + 1e-12),
               extent=[-2,2,-2,2],
               origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(label="Log Probability")
    plt.title(f"Langevin Dist (Chunked) x_mean={mean_x:.2f},y_mean={mean_y:.2f}")
    plt.savefig("langevin_distribution_heavy.png", dpi=300)
    plt.close()
    print("Saved langevin_distribution_heavy.png")

    # FOkker Planck
    print("\nFOKKER-PLANCK PDE")

    Nx, Ny = 50, 50
    dt_fp = 1e-5
    T_fp  = 0.005
    x_grid, y_grid, P_final, conv_data = solve_fokker_planck_2d(
        D=0.2,
        x_min=-2.0, x_max=2.0, Nx=Nx,
        y_min=-2.0, y_max=2.0, Ny=Ny,
        dt=dt_fp, T=T_fp
    )

    print(f"DEBUG: PDE final => shape={P_final.shape}, "
          f"min={float(P_final.min())}, max={float(P_final.max())}, sum={float(P_final.sum())}")

    plt.figure(figsize=(7,6))
    plt.imshow(jnp.log(P_final + 1e-12),
               extent=[-2,2,-2,2],
               origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label="Log Probability")
    plt.title("Fokker–Planck Dist (Reduced, log-scale)")
    plt.savefig("fokker_planck_distribution_heavy.png", dpi=300)
    plt.close()
    print("Saved fokker_planck_distribution_heavy.png")

    # PDE Convergence
    steps_array = np.arange(conv_data.shape[0])
    conv_data_np = np.array(conv_data)
    plt.figure(figsize=(7,5))
    plt.plot(steps_array, conv_data_np, 'b-')
    plt.yscale('log')
    plt.xlabel("Time Step")
    plt.ylabel("L1 Diff (P_new - P_old)")
    plt.title("Fokker–Planck PDE Convergence (Reduced, log-scale)")
    plt.grid(True)
    plt.savefig("pde_convergence_heavy.png", dpi=300)
    plt.close()
    print("Saved pde_convergence_heavy.png")

    overall_end = time.time()
    print(f"\n Time: ~{(overall_end - overall_start):.2f}s.")
    print("Generated files:\n"
          "  1) potential_landscape_heavy.png\n"
          "  2) langevin_distribution_heavy.png\n"
          "  3) fokker_planck_distribution_heavy.png\n"
          "  4) pde_convergence_heavy.png\n")

if __name__ == "__main__":
    main()
