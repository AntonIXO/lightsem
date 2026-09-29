"""Event-driven progress notification; no polling, fitting, or GPU use."""
import argparse, ctypes, json, os, selectors
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--pid',type=int,required=True)
p.add_argument('--progress',type=Path,required=True)
p.add_argument('--completed',type=int,required=True)
a=p.parse_args()
libc=ctypes.CDLL(None,use_errno=True)
libc.inotify_init1.argtypes=[ctypes.c_int]
libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
libc.pidfd_open.argtypes=[ctypes.c_int,ctypes.c_uint]
fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
if fd<0: raise OSError(ctypes.get_errno(),'inotify_init1')
if libc.inotify_add_watch(fd,os.fsencode(a.progress.parent),0x8|0x80|0x100)<0:
    raise OSError(ctypes.get_errno(),'inotify watch')
pidfd=libc.pidfd_open(a.pid,0)
sel=selectors.DefaultSelector(); sel.register(fd,selectors.EVENT_READ,'files')
if pidfd>=0: sel.register(pidfd,selectors.EVENT_READ,'process')
def report(ended=False):
    rows=[]
    if a.progress.exists():
        for line in a.progress.read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if row.get('event')=='job_complete': rows.append(row)
    if len(rows)>=a.completed or ended:
        print(json.dumps({'status':'milestone' if len(rows)>=a.completed else 'queue_ended_before_milestone',
            'requested_completed':a.completed,'completed':len(rows),'progress':str(a.progress),
            'last_completed':rows[-1] if rows else None}),flush=True)
        return True
    return False
if not report(ended=pidfd<0):
    done=False
    while not done:
        for key,_ in sel.select():
            if key.data=='files':
                os.read(fd,65536)
                done=report()
            else:
                done=report(ended=True)
            if done: break
sel.close(); os.close(fd)
if pidfd>=0: os.close(pidfd)
