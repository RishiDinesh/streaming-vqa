import glob
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import ImageDraw, ImageFont

from .base import BaseVideoQADataset

SECRET_WORDS = [
    # Names
    "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Helen",
    "Ivan", "Jack", "Kate", "Leo", "Mike", "Nick", "Oscar", "Paul",
    "Quinn", "Rose", "Sam", "Tom", "Uma", "Vince", "Wendy", "Xena",
    "Yvonne", "Zack", "Amy", "Ben", "Clara", "Dan", "Emma", "Fred",
    "Gina", "Henry", "Iris", "Jade", "Kyle", "Lucy", "Max", "Nina",
    "Owen", "Pam", "Ray", "Sara", "Tina", "Uri", "Vera", "Walt",
    "Yuri", "Zoe", "Amber", "Blake", "Chloe", "Derek", "Elena", "Felix",
    "Hana", "Igor", "Jules", "Kira", "Liam", "Maya", "Noah", "Olive",
    "Petra", "Reese", "Stella", "Theo", "Vivian", "Wyatt", "Axel", "Briar",
    "Cyrus", "Diana", "Elton", "Flora", "Grant", "Holly", "Jasper", "Kenji",
    "Luna", "Marco", "Nadia", "Orion", "Penny", "Rufus", "Sage", "Tobias",
    "Ursa", "Violet", "Wren", "Xavier", "Yara", "Zelda", "Arlo", "Blythe",
    "Cruz", "Daphne", "Ezra", "Fern",
    # Animals
    "Falcon", "Tiger", "Dolphin", "Eagle", "Panda", "Wolf", "Raven", "Cobra",
    "Otter", "Jaguar", "Crane", "Bison", "Gecko", "Heron", "Koala", "Lemur",
    "Manta", "Newt", "Okapi", "Parrot", "Quail", "Robin", "Shark", "Toucan",
    "Viper", "Walrus", "Yak", "Zebra", "Alpaca", "Badger", "Condor", "Dingo",
    "Ermine", "Ferret", "Gorilla", "Hyena", "Iguana", "Jackal", "Kiwi", "Lynx",
    "Moose", "Narwhal", "Osprey", "Puffin", "Quetzal", "Raptor", "Stork", "Tapir",
    "Urchin", "Vulture", "Wombat", "Xerus", "Giraffe", "Pelican", "Mantis", "Beetle",
    "Coyote", "Donkey", "Finch", "Grouse", "Hornet", "Impala", "Jacana", "Kestrel",
    "Lobster", "Marmot", "Numbat", "Oriole", "Pigeon", "Rabbit", "Salmon", "Turtle",
    "Vervet", "Weasel", "Xenops", "Gopher", "Hermit", "Ibis", "Jerboa", "Lark",
    # Objects
    "Anchor", "Basket", "Candle", "Dagger", "Engine", "Flagon", "Goblet", "Hammer",
    "Inkwell", "Jacket", "Kettle", "Lantern", "Mirror", "Needle", "Obelisk", "Pillar",
    "Quiver", "Ribbon", "Scepter", "Trumpet", "Umbrella", "Vessel", "Widget", "Zipper",
    "Anvil", "Beacon", "Chisel", "Drum", "Easel", "Funnel", "Garnet", "Hatchet",
    "Ivory", "Jewel", "Knapsack", "Locket", "Mortar", "Nugget", "Orchid", "Prism",
    "Quartz", "Ratchet", "Saddle", "Thimble", "Urn", "Vial", "Wrench", "Xylophone",
    "Buckle", "Compass", "Decanter", "Emblem", "Feather", "Gauntlet", "Hourglass",
    "Insignia", "Javelin", "Keystone", "Lattice", "Medallion", "Nozzle", "Pendant",
    "Relic", "Spindle", "Talisman", "Utensil", "Valve", "Wagon", "Abacus",
    "Bellows", "Caliper", "Dowel", "Eyelet", "Flint", "Gimbal", "Harness", "Ingot",
    # Nature
    "Aurora", "Blizzard", "Canyon", "Delta", "Eclipse", "Fjord", "Glacier", "Horizon",
    "Island", "Jungle", "Karst", "Lagoon", "Monsoon", "Nebula", "Oasis", "Prairie",
    "Quasar", "Ravine", "Summit", "Tsunami", "Volcano", "Wetland", "Zenith", "Bamboo",
    "Cedar", "Daisy", "Elm", "Ficus", "Grove", "Hazel", "Ivy", "Juniper",
    "Kelp", "Lotus", "Maple", "Nettle", "Oak", "Palm", "Reed", "Spruce",
    "Thicket", "Tundra", "Willow", "Acacia", "Birch", "Clover", "Dune", "Estuary",
    "Geyser", "Heath", "Inlet", "Jasmine", "Kindle", "Lichen", "Moss",
    "Marigold", "Poplar", "Redwood", "Savanna", "Terrace",
    # Foods
    "Almond", "Biscuit", "Cashew", "Dumpling", "Espresso", "Fondue", "Granola",
    "Hazelnut", "Icing", "Jambalaya", "Kumquat", "Lychee", "Mango", "Nougat",
    "Pretzel", "Quinoa", "Raisin", "Sorbet", "Truffle", "Vanilla", "Waffle",
    "Apricot", "Brioche", "Clementine", "Focaccia", "Ginger", "Hummus",
    "Kale", "Lemon", "Muffin", "Nutmeg", "Papaya", "Rhubarb", "Saffron", "Tapioca",
    "Wasabi", "Arugula", "Basil", "Cinnamon", "Dill", "Fennel", "Garlic", "Honey",
    "Jalapeno", "Lavender", "Mint", "Pepper", "Sesame", "Thyme", "Turnip",
    # Science / Abstract
    "Atom", "Binary", "Cipher", "Dynamo", "Electron", "Fractal", "Genome", "Helix",
    "Isotope", "Joule", "Kelvin", "Lambda", "Matrix", "Neutron", "Omega", "Photon",
    "Quantum", "Reactor", "Sigma", "Tensor", "Uranium", "Vector", "Waveform", "Xenon",
    "Alpha", "Beta", "Carbon", "Doppler", "Entropy", "Flux", "Gamma", "Hadron",
    "Inertia", "Kinetic", "Muon", "Nucleus", "Orbit", "Plasma", "Quarks",
    "Radium", "Syntax", "Theorem", "Upsilon", "Vertex", "Wavelength", "Axiom", "Boson",
    "Cosine", "Decibel", "Epsilon", "Faraday", "Gauss", "Hertz", "Impulse", "Kinase",
    "Laser", "Magnet", "Newton", "Optics", "Proton", "Reflex", "Scalar", "Torque",
    "Unity", "Vortex", "Weber", "Ampere", "Boron", "Cobalt", "Diode", "Ether",
    "Fusion", "Gravity", "Hybrid", "Iodine", "Krypton", "Lithium", "Manganese", "Neon",
]

SECRET_WORDS = list(dict.fromkeys(SECRET_WORDS))

ORDINALS = [
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
]


def _load_vnbench_font(height):
    font_size = max(14, int(height * 0.06))
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(base_dir, "fonts", "OpenSans.ttf")
    try:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    except Exception:
        pass
    return ImageFont.load_default()


def burn_subtitle_vnbench(frame_img, text, font):
    w, h = frame_img.size
    draw = ImageDraw.Draw(frame_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_y = max(4, int(text_h * 0.30))
    bar_h = text_h + 2 * pad_y
    bar_top = int(h * 0.85) - bar_h
    bar_bottom = bar_top + bar_h

    draw.rectangle([0, bar_top, w, bar_bottom], fill=(80, 80, 80))

    text_x = (w - text_w) // 2 - bbox[0]
    text_y = bar_top + pad_y - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
    return frame_img


class DynamicSyntheticVideoQADataset(BaseVideoQADataset):
    def __init__(
        self,
        video_root: str,
        processor: Optional[Any] = None,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        num_frames: int = 64,
        max_length: int = 32768,
        use_chat_template: bool = True,
        num_needles: int = 5,
        min_depth_ratio: float = 0.2,
        max_depth_ratio: float = 0.8,
        dataset_len: int = 50000,
        frame_idx: Optional[List[int]] = None,
    ):
        super().__init__(
            video_root=video_root,
            processor=processor,
            model_id=model_id,
            num_frames=num_frames,
            max_length=max_length,
            use_chat_template=use_chat_template
        )

        if num_needles <= 0:
            raise ValueError("`num_needles` must be > 0.")
        if dataset_len <= 0:
            raise ValueError("`dataset_len` must be > 0.")

        self.num_needles = int(num_needles)
        self.min_depth_ratio = float(min_depth_ratio)
        self.max_depth_ratio = float(max_depth_ratio)
        self.frame_idx = frame_idx
        self._dataset_length = int(dataset_len)

        self.video_files: List[str] = []
        for ext in ["*.mp4", "*.webm", "*.avi", "*.mkv"]:
            pattern = os.path.join(self.video_root, "**", ext)
            for fp in glob.glob(pattern, recursive=True):
                fname = os.path.basename(fp).lower()
                if any(
                    x in fname
                    for x in ["_cnt_", "_ord_", "_ret_", "_synth", "ret_edit"]
                ):
                    continue
                self.video_files.append(os.path.abspath(fp))

        if not self.video_files:
            raise ValueError(
                f"No valid unedited source videos found in {self.video_root}."
            )

    def _build_sample(self, index: int) -> Dict[str, Any]:
        video_path = self.video_files[index % len(self.video_files)]
        frames = self._decode_and_sample_frames(video_path)

        chosen_words = random.sample(SECRET_WORDS, self.num_needles)

        if self.frame_idx is not None and len(self.frame_idx) == self.num_needles:
            mapped_indices = sorted(self.frame_idx)
            mapped_indices = [
                min(max(0, idx), max(0, len(frames) - 1)) for idx in mapped_indices
            ]
        else:
            num_intervals = 20
            intervals = np.linspace(
                self.min_depth_ratio, self.max_depth_ratio, num_intervals
            )
            chosen_indices = np.random.choice(
                len(intervals), size=self.num_needles, replace=False
            )
            depth_ratios = np.sort(intervals[chosen_indices]).tolist()
            mapped_indices = [
                int(round(ratio * (len(frames) - 1))) for ratio in depth_ratios
            ]

        if len(frames) > 0:
            height = frames[0].height
            font = _load_vnbench_font(height)

            for i, (word, mapped_idx) in enumerate(zip(chosen_words, mapped_indices)):
                target_frame = frames[mapped_idx]

                ordinal = ORDINALS[i] if i < len(ORDINALS) else f"#{i + 1}"
                subtitle = f"The {ordinal} secret word is: {word}"

                frames[mapped_idx] = burn_subtitle_vnbench(target_frame, subtitle, font)

        if self.num_needles == 1:
            question = "What is the first secret word based on the given video:\n"
        else:
            question = (
                f"What are the {self.num_needles} secret words based on the given video:\n"
            )

        answer_lines = []
        for i, word in enumerate(chosen_words):
            ordinal = ORDINALS[i] if i < len(ORDINALS) else f"#{i + 1}"
            answer_lines.append(f"The {ordinal} secret word is: {word}")
        supervision_text = "\n".join(answer_lines)

        # Prompt ends at the question; supervision starts with the first answer line.
        prefix_text = self._build_prefix_text(question)
        full_text = f"{prefix_text}{supervision_text}"

        return {
            "frames": frames,
            "prefix_text": prefix_text,
            "full_text": full_text,
        }
