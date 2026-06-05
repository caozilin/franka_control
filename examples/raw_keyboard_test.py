#!/usr/bin/env python3

from __future__ import annotations

import select
import sys
import termios
import tty


class RawKeyboardReader:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def poll(self):
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            return sys.stdin.read(1)
        return None


def main() -> int:
    print("raw keyboard test")
    print("focus this terminal and press keys; ESC exits")
    with RawKeyboardReader() as reader:
        while True:
            char = reader.poll()
            if char is None:
                continue
            print(f"got: {repr(char)}")
            if char == "\x1b":
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
