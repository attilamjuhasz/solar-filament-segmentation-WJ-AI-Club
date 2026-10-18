import ctypes
import sys
import winsound

title = sys.argv[1] if len(sys.argv) > 1 else "Project Helios"
message = sys.argv[2] if len(sys.argv) > 2 else "Task completed!"
status = sys.argv[3] if len(sys.argv) > 3 else "success"

is_error = status.lower() in ("error", "failure")

icon = 0x30 if is_error else 0x40
flags = 0x40000 | 0x10000 | icon

winsound.MessageBeep(0x30 if is_error else 0x40)

ctypes.windll.user32.MessageBoxW(
    None,
    message,
    title,
    flags
)
