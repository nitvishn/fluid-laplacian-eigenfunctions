This is an implementation of [Fluid Simulation using Laplacian Eigenfunctions](https://dl.acm.org/doi/10.1145/2077341.2077351). To test:

```
pip install -r requirements.txt
python main.py
```

My simulation takes place on the domain $[0, \pi]^2$, representing the fluid in a finite basis of Laplacian eigenfunctions. I use the analytical expressions of the eigenfunctions on this domain. 

Given some initial velocity field $u$, the code computes $w$, the coefficients of the projection of $u$ onto the eigenfunction basis. The code evolves $w$ as specified in the paper, and visualizes the solution via particles. 

Note that the simulation _almost_ does not see the grid and the particles, they are for projection and visualization. I say almost, because there is one exception to this. The implementation observes the particle velocities to inform adaptive rescaling of the timestep. I added this feature so that the particles don't jump too far when we simulate high velocity flows. Although the simulation is unconditionally stable, this does help the results look reasonable. 

Most of the run time is spent advecting the particles, of which I use tens of thousands. The simulation itself is some orders of magnitude faster than this implementation would suggest. This is seen by setting the number of particles to like, 100..

There's no data required to run the simulation, simply install the required packages and run `main.py`. I developed this on a MacBook with an M1 Pro chip. I used PyTorch thinking I would support all operations on the GPU, but I didn't get around to this in time. 

The output video will be in `sim_id/sim_id_particles.mp4`, and the current frame is maintained in `sim_id/sim_id_current.png` if you'd like to monitor the output. 

I've attached 3 video files in `extras/`, all at resolution=512x512:
- `res_512_n_9.mp4`: 9 eigenfunctions
- `res_512_N_49.mp4`: 49 eigenfunctions
- `res_512_N_196.mp4`: 196 eigenfunctions

I enjoyed this course very much, have a wonderful winter break!