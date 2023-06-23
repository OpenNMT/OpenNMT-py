from tensorboard.backend.event_processing import event_accumulator
from argparse import ArgumentParser
import os


class TestEvents:
    def __init__(self):
        stats = ["xent", "ppl", "accuracy", "tgtper", "lr"]
        metrics = ["BLEU", "TER"]
        self.scalars = {}
        self.scalars["train"] = [("progress/" + stat) for stat in stats]
        self.scalars["valid"] = [("valid/" + stat) for stat in stats]
        self.scalars["valid_metrics"] = self.scalars["valid"] + [
            ("valid/" + metric) for metric in metrics
        ]

    def reload_events(self, path):
        ea = event_accumulator.EventAccumulator(
            path,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
        return ea

    def check_scalars(self, scalars, logdir):
        for event_file in os.listdir(logdir):
            path = os.path.join(logdir, event_file)
            event_accumulator = self.reload_events(path)
            # make sure the scalars are in the event accumulator tags
            assert all(
                s in event_accumulator.Tags()["scalars"] for s in scalars
            ), "{} some scalars were not found in the event accumulator"


if __name__ == "__main__":
    # required arguments
    parser = ArgumentParser()
    requiredArgs = parser.add_argument_group("required arguments")
    requiredArgs.add_argument("-logdir", "--logdir", type=str, required=True)
    requiredArgs.add_argument(
        "-tensorboard_checks",
        "--tensorboard_checks",
        type=str,
        required=True,
        choices=["train", "valid", "valid_metrics"],
    )
    args = parser.parse_args()
    test_event = TestEvents()
    scalars = test_event.scalars[args.tensorboard_checks]
    print("looking for scalars: ", scalars)
    test_event.check_scalars(scalars, args.logdir)
