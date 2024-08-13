from .audio_classes import MultiTrack, Source, Target
import os
import typing as tp
from pathlib import Path
import collections
from torch.utils.data import Dataset
import yaml

# import open_dataset
class OpenDataset(Dataset):
    def __init__(
        self,
        root: str,
        setup_file: str = None,
        is_wav: bool = False,
        subsets: tp.List[str] = ['train', 'test'],
        split: tp.Optional[str] = 'train',
        sample_rate: int = None,
    ):
        self.root = Path(root)

        if setup_file is not None:
            setup_path = Path(self.root, setup_file)
        else:
            setup_path = os.path.join(
                Path(__file__).parent.absolute(), 'configs', 'open.yaml'
            )
        
        with open(setup_path, 'r') as f:
            self.setup = yaml.safe_load(f)
        
        if sample_rate != self.setup['sample_rate']:
            self.sample_rate = sample_rate
        self.sources_names = list(self.setup['sources'].keys())
        self.is_wav = is_wav
        
        self.tracks = self.load_open_tracks(subsets=subsets, split=split)
    
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, index):
        return self.tracks[index]
    
    def load_open_tracks(self, subsets=None, split=None):
        if subsets is not None:
            if isinstance(subsets, str):
                subsets = [subsets]
        else:
            subsets = ['train', 'test']

        if subsets != ['train'] and split is not None:
            raise RuntimeError("Subset has to set to `train` when split is used")
        
        tracks = []
        for subset in subsets:
            subset_folder = Path(self.root, subset)
            
            for _, folders, files in os.walk(subset_folder):
                for track_name in sorted(folders):
                    if subset == 'train':
                        if split == 'train' and track_name in self.setup['validation_tracks']:
                            continue
                        elif split == 'valid' and track_name not in self.setup['validation_tracks']:
                            continue

                    track_folder = Path(subset_folder, track_name)
                    # create new mus track
                    track = MultiTrack(
                        name=track_name,
                        path=Path(track_folder, self.setup['mixture']),
                        # path=op.join(
                        #     track_folder,
                        #     self.setup['mixture']
                        # ),
                        subset=subset,
                        is_wav=self.is_wav,
                        stem_id=self.setup['stem_ids']['mixture'],
                        sample_rate=self.sample_rate
                    )

                    # add sources to track
                    sources = {}
                    for src, source_file in list(
                        self.setup['sources'].items()
                    ):
                        # create source object
                        abs_path = Path(track_folder, source_file)
                        # abs_path = op.join(
                        #     track_folder,
                        #     source_file
                        # )
                        if os.path.exists(abs_path):
                            sources[src] = Source(
                                track,
                                name=src,
                                path=abs_path,
                                stem_id=self.setup['stem_ids'][src],
                                sample_rate=self.sample_rate
                            )
                    track.sources = sources
                    track.targets = self.create_targets(track)

                    # add track to list of tracks
                    tracks.append(track)
        print(tracks)
        return tracks
    
    def create_targets(self, track):
        # add targets to track
        targets = collections.OrderedDict()
        for name, target_srcs in list(
            self.setup['targets'].items()
        ):
            # add a list of target sources
            target_sources = []
            for source, gain in list(target_srcs.items()):
                if source in list(track.sources.keys()):
                    # add gain to source tracks
                    track.sources[source].gain = float(gain)
                    # add tracks to components
                    target_sources.append(track.sources[source])
                    # add sources to target
            if target_sources:
                targets[name] = Target(
                    track,
                    sources=target_sources,
                    name=name
                )

        return targets