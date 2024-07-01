import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import os
import time as tm

out_folder = "C:\\Users\\Tripp\\Desktop\\VSC Projects\\rw_gas"

x_min=0
x_max=2
y_min=-1
y_max=1
step_max=0.01
DRIFT=0.005
xsize=10000
tot_iters=1000
particle_count=1000
collision_radius = 0.01

def clear():
    os.system('cls')
    
# this is a vector field!
def drift(x,y):
    Vx = DRIFT #0.1-y**2
    Vy = 0 #(x-1.1)*y
    return [Vx, Vy]
# look into passing fns as args

def collide():

    return False

def step(x, y, max_step=step_max, drift_bool=False):
    nx = 0
    xrand = np.random.uniform(-max_step, max_step, len(x))
    yrand = np.random.uniform(-max_step, max_step, len(x))
    
    for i in range(len(x)):
        if x[i] != 2:
            # apply random walk + drift
            drifts = drift(x,y) if drift_bool else [0,0]
            x[i] += xrand[i] + drifts[0]
            y[i] += yrand[i] + drifts[1]

            # x boundary conditions
            if x[i] < x_min:
                x[i] = x_min
            elif x[i] > x_max:
                x[i] = x_max
                nx+=1

            # y boundary conditions
            if y[i] < y_min:
                y[i] = y_min
            elif y[i] > y_max:
                y[i] = y_max

    return x, y, nx

def run(x, y, iters, drift_bool=False):
    currentTime = tm.perf_counter()
    counter = 0
    Nval = len(x)
    steps = np.linspace(0, iters, iters+1)
    N_vals = np.zeros(iters+1)
    xnew = x
    ynew = y

    data = [[x, y]] # optimize?  [(np.zeros(xsize), np.zeros(xsize)) for i in range(iters+1)]
    print(f"N: {Nval}")
    # do a step
    while(counter < iters):
        xnew, ynew, count_new = step(x, y, drift_bool=drift_bool)

        Nval -= count_new
        N_vals[counter] = Nval

        x = xnew
        y = ynew
        
        # update overall data
        data.append(np.array((x,y)))
        counter +=1

    # USE TO END SIMULATION EARLY
    '''
        if Nval == 0:
            break
    
    if len(data) != len(steps):
        steps = steps[:(len(data))]
        N_vals = N_vals[:(len(data))]
    '''

    time = np.round(tm.perf_counter() - currentTime, 5)
    print(f"{counter} iterations in {time} seconds, {np.round(counter/time, 1)} iterations/second")
    return steps, N_vals, data, time

def plot(time1=0,time2=0):
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.set_figwidth(15)
    fig.set_figheight(8)

    scat1 = ax1.scatter([],[])
    scat2 = ax1.scatter([],[])
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.tick_params(
        bottom=False, top=False, left=False, right=False,
        labelbottom=False,labelleft=False,
        labelright=False, labeltop=False
    )

    trace1, = ax2.plot([], [], '.-', lw=0.5, ms=2, label="No Drift")
    trace2, = ax2.plot([], [], '.-', lw=0.5, ms=2, label="Drift")
    ax2.set_xlim([0, tot_iters+1])
    ax2.set_ylim([0, xsize])
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("# of Particles in Box")
    
    ax2.legend()
    ax1.legend([scat1, scat2], ["Population 1", "Population 2"], loc="center", bbox_to_anchor=[0.5,-0.1], ncol=2)

    plt.suptitle(f"Random Walk, N={xsize}\nPop. 1 Sim. Time: {time1}s\n Pop. 2 Sim. Time: {time2}s")

    return fig, scat1, scat2, trace1, trace2

def animate(i):
    hist_x=steps1[:i]
    hist_N=N1[:i]

    hist_x2=steps2[:i]
    hist_N2=N2[:i]

    trace1.set_data(hist_x, hist_N)
    trace2.set_data(hist_x2, hist_N2)
    # data = [ (x(0), y(0)), (x(1), y(1)), ... , (x(N), y(N))]
    scat1.set_offsets(np.column_stack([data1[i][0][:particle_count], data1[i][1][:particle_count]]))
    scat2.set_offsets(np.column_stack([data2[i][0][:particle_count], data2[i][1][:particle_count]]))

    return trace1, scat1, trace2, scat2,

clear()

# random, initial positions
x_start1 = np.random.uniform(x_min, x_max, xsize)
y_start1 = np.random.uniform(y_min, y_max, xsize)
x_start2 = np.random.uniform(x_min, x_max, xsize)
y_start2 = np.random.uniform(y_min, y_max, xsize)

# run simulation
steps1, N1, data1, time1 = run(x_start1, y_start1, tot_iters)
steps2, N2, data2, time2 = run(x_start2, y_start2, tot_iters, drift_bool=True)

# run animation
fig, scat1, scat2, trace1, trace2 = plot(time1=time1, time2=time2)

ani = anim.FuncAnimation(
    fig, animate, tot_iters, interval=10, blit=True, repeat=False
)

# save animation as a gif using PillowWriter
'''
writer = anim.PillowWriter(fps=15,
    metadata=dict(artist='Me'),
    bitrate=1800)
print("Saving...")
sci_xsize = "{:1e}".format(xsize)
sci_iters = "{:1e}".format(tot_iters)
fname = f"v1_linear-drift_{DRIFT}"
ani.save(os.path.join(out_folder, fname+".gif"), writer=writer)
print("Saved!")
'''

plt.show()

# CODE FOR REAL-TIME
'''
def frames():
    while True:
        yield func

# where func is step function
'''
