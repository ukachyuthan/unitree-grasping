# Settings for experiments on the real-scale object set (source this).
#   data/objects_v2   real-size YCB, concavities kept (generate_ycb_meshes.py --real_scale --keep_concave)
#   table clearance 0 the fingertips may touch and slide along the table
#   contact bound     ±12 cm: at ±7 cm, 29 real-size objects had no reachable GraspNet grasp
export GRASP_OBJECTS_DIR=data/objects_v2
export GRASP_TABLE_CLEARANCE_M=0
export GRASP_CONTACT_BOUND_M=0.12
export PAPER_OUT=../../data/paper_v2
