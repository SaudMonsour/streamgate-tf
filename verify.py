"""Fresh-source hashes and independent checkpoint replay of published scores."""
import json
from pathlib import Path
import numpy as np
from train import evaluate
from streamgate.data import load_corpus, windows, digest
from streamgate.model import ByteDecoder, DecoderConfig


def main():
    (_, validation, test), manifest = load_corpus(verify_remote=True)
    recorded = json.loads(Path("metrics.json").read_text())
    audit = json.loads(Path("audit.json").read_text())
    for path, expected in audit["files_sha256"].items():
        assert digest(Path(path).read_bytes()) == expected, path
    checks = []
    for kind in ("softmax", "linear"):
        model = ByteDecoder(DecoderConfig(**recorded["models"][kind]["config"]))
        model.load_npz(f"checkpoints/{kind}.npz")
        for name, data in (("validation", validation), ("holdout", test)):
            x, y, _ = windows(data, recorded["config"]["context"])
            replay, losses, predictions = evaluate(model, x, y)
            for metric, value in recorded["models"][kind][name].items():
                np.testing.assert_allclose(replay[metric], value, rtol=1e-6, atol=1e-7)
            checks.append({"model": kind, "split": name, "replay": replay})
            if name == "holdout":
                with np.load(f"runs/{kind}_holdout.npz", allow_pickle=False) as saved:
                    np.testing.assert_array_equal(y, saved["targets"])
                    np.testing.assert_array_equal(predictions, saved["predictions"])
                    np.testing.assert_allclose(losses, saved["losses"], atol=1e-6)
    result = {"status": "passed", "source_sha256": manifest["source_sha256"],
              "fresh_download_matches_every_source_byte": True,
              "checkpoint_hashes_match_audit": True, "checks": checks}
    Path("runs/verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
