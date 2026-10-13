# Byte guard newline mismatch

The verified-resume attempt stopped before further material copies because the original write_text used Windows CRLF while the resume renderer expected LF bytes. Direct comparison confirmed all 113 newline differences were formatting only and the JSON content was identical. The guard remained strict; it was repaired to render the original platform newline representation, preserving every existing artifact byte. No scientific result changed and no simulator or fit ran.
