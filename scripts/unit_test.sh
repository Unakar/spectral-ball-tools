python -u root_solver.py \
  --method fixed_point \
  --n 4096 \
  --m 128 \
  --seed 42 \
  --tol 1e-4 \
  --max_iter 100 \
  --msign_steps 10 \
  --theta_source svd \
  --power_iters 30