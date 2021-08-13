import sys
import pickle

sys.path.append('/zfs/yitians/darkhistory/DarkHistory/')
import main

SAVE_DIR = '/zfs/yitians/darkhistory/DarkHistory/nntf/tests/run_output/'

def save_fn(config):
    fn =  'E8_'
    fn += str(config['start_rs']) + '_'
    fn += str(config['end_rs']) + '_c'
    fn += str(config['coarsen_factor']) + '_'
    fn += str(config['tf_mode']) + '.solcfg' # soln config
    return fn


run_config = {}

run_config['DM_process']   = 'decay'
run_config['mDM']          = 1e8
run_config['lifetime']     = 3e25
run_config['primary']      = 'elec_delta'
run_config['backreaction'] = True
run_config['helium_TLA']   = True
run_config['reion_switch'] = True

run_config['start_rs']       = 3000
run_config['end_rs']         = 4
run_config['coarsen_factor'] = 12
run_config['tf_mode']        = ('table' if len(sys.argv)==1 else sys.argv[1])

save_fn(run_config)
print('\nSaving to '+save_fn(run_config))

if input('Continue [y/n]: ') != 'y':
    exit()

soln = main.evolve(**run_config)

pickle.dump( (soln, run_config), open(SAVE_DIR+save_fn(run_config), 'wb') )