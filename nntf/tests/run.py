import sys
import pickle

sys.path.append('/zfs/yitians/darkhistory/DarkHistory/')
import main

SAVE_DIR = '/zfs/yitians/darkhistory/DarkHistory/nntf/tests/run_output/'

def save_fn(config, comment=None):
    fn =  'E8ph'
    fn += '_' + str(config['start_rs'])
    fn += '_' + str(config['end_rs'])
    fn += '_c' + str(config['coarsen_factor'])
    fn += '_' + str(config['tf_mode'])
    if comment is not None:
        fn += '_' + str(comment)
    fn += '.solcfg' # soln config
    return fn


run_config = {}

run_config['DM_process']   = 'decay'
run_config['mDM']          = 1e8
run_config['lifetime']     = 3e25
run_config['primary']      = 'phot_delta'
run_config['backreaction'] = True
run_config['helium_TLA']   = True
run_config['reion_switch'] = True

run_config['start_rs']       = 3000
run_config['end_rs']         = 4
run_config['coarsen_factor'] = 12
run_config['tf_mode']        = ('table' if len(sys.argv)==1 else sys.argv[1])

comment = None if len(sys.argv)<=2 else sys.argv[2]


print('\nSaving to '+save_fn(run_config, comment))

if input('Continue (y/n): ') != 'y':
    exit()

soln = main.evolve(**run_config)

pickle.dump( (soln, run_config), open(SAVE_DIR+save_fn(run_config, comment), 'wb') )