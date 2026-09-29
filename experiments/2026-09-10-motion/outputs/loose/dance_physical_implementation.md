# Physical loss implementation: parent recovery

The worker wrote `music_physical.py` and `test_music_physical.py`, then its provider failed before the final report. The parent read the implementation and reran all15 tests through `CUDA_VISIBLE_DEVICES="" bash run_torch.sh -m unittest discover -s tests -p test_music_physical.py -v`: all pass (0.461s test execution).

Opt-in only; no model loss or defaults changed. `clean_estimate` implements x_t+(1-t)*v in FP32. `physical_losses` exposes separate FK position/velocity, reference-contact planted-foot velocity, root-displacement and floor-penetration losses. Masks exclude padding and boundary derivatives, support short holds and reset root origins across gaps. Generated contacts cannot hide sliding. No combined weight selection or trained improvement is claimed. GPU preflight, real-example gradients, loss weights, experiment registration and trained evaluation are still required before integration.
