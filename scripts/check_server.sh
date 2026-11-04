#!/bin/bash
# Start / stop / restart the live robot check for the VR app (grasp_check_server.py).
#
#   scripts/check_server.sh start [cases_root]   default cases_root: data/vr_round3 (object set v2)
#   scripts/check_server.sh stop
#   scripts/check_server.sh restart [cases_root]
#
# The pid is kept in data/check_server.pid, so stopping never pattern-matches processes.
cd "$(dirname "$0")/.."
PIDFILE=data/check_server.pid
LOG=data/paper/logs/check_server.log

stop() {
  [ -f $PIDFILE ] && kill -9 "$(cat $PIDFILE)" 2>/dev/null
  rm -f $PIDFILE
}

start() {
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate unitree_isaaclab
  export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes
  source scripts/v2_env.sh
  (cd wrappers/isaaclab && exec python scripts/grasp_check_server.py --headless \
     --cases_root "../../${1:-data/vr_round3}") > $LOG 2>&1 &
  echo $! > $PIDFILE
  for _ in $(seq 60); do
    curl -s -m 2 http://127.0.0.1:8765/health && { echo; return 0; }
    sleep 5
  done
  echo "check server did not come up; see $LOG"; return 1
}

case "$1" in
  start) start "$2" ;;
  stop) stop ;;
  restart) stop; sleep 2; start "$2" ;;
  *) echo "usage: $0 start|stop|restart [cases_root]"; exit 2 ;;
esac
