"""Music theory in a few lines. No libraries needed.

A chord = a ROOT note + other notes a fixed number of SEMITONES above it.
(A semitone is one piano key, black keys included.)

    major   : root, +4, +7
    minor   : root, +3, +7
    seventh : root, +4, +7, +10      (a "dominant 7th")

Frequency of a note that is s semitones above a root:  root_hz * 2 ** (s / 12)
"""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_PC = {"A": 9, "B": 11, "C": 0, "D": 2, "E": 4, "F": 5, "G": 7}   # pitch class of each letter
ROOT_HZ = {"A": 220.00, "B": 246.94, "C": 261.63, "D": 293.66, "E": 329.63, "F": 349.23, "G": 392.00}
INTERVALS = {"major": (0, 4, 7), "minor": (0, 3, 7), "seventh": (0, 4, 7, 10)}
QUALITY_LABEL = {"major": "major", "minor": "minor", "seventh": "7th"}


def chord_freqs(letter: str, quality: str) -> list[float]:
    return [ROOT_HZ[letter] * 2 ** (s / 12) for s in INTERVALS[quality]]


def chord_note_names(letter: str, quality: str) -> list[str]:
    return [NOTE_NAMES[(ROOT_PC[letter] + s) % 12] for s in INTERVALS[quality]]
