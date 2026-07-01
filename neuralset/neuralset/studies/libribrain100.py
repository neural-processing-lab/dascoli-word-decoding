# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from ..data import BaseData


def _decode_attr(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    return [
        item.decode() if isinstance(item, bytes) else str(item)
        for item in np.asarray(value).tolist()
    ]


def _column_or_default(df: pd.DataFrame, column: str, default) -> pd.Series:
    if column in df:
        return df[column]
    return pd.Series(default, index=df.index)


class LibriBrain100(BaseData):
    """PNPL LibriBrain100 adapter.

    Each PNPL ``(subject, session, task, run)`` record is exposed as one
    neuralset timeline. PNPL handles file resolution/download; neuralset
    receives an event table with a ``Meg`` data event and ``Word`` triggers.
    """

    subject: str
    session: str
    task: str
    run: str
    corpus: str
    partition: str

    url: tp.ClassVar[str] = "https://github.com/neural-processing-lab/pnpl"
    licence: tp.ClassVar[str] = "TODO"
    device: tp.ClassVar[str] = "Meg"
    description: tp.ClassVar[
        str
    ] = "LibriBrain100 MEG recordings loaded through the PNPL package."
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("pnpl", "h5py")

    @classmethod
    def _download(cls, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _iter_timelines(cls, path: str | Path) -> tp.Iterator["LibriBrain100"]:
        try:
            from pnpl.datasets.libribrain100 import RUN_RECORDS
        except ImportError as exc:
            raise ImportError(
                "LibriBrain100 requires the PNPL package. Install it with `pip install pnpl`."
            ) from exc

        for record in RUN_RECORDS:
            yield cls(
                subject=record.subject,
                session=record.session,
                task=record.task,
                run=record.run,
                corpus=record.corpus,
                partition=record.partition,
                path=path,
            )

    def _pnpl_dataset(self):
        try:
            from pnpl.datasets import LibriBrain100 as PNPLLibriBrain100
            from pnpl.tasks.libribrain import WordClassification
        except ImportError as exc:
            raise ImportError(
                "LibriBrain100 requires the PNPL package. Install it with `pip install pnpl`."
            ) from exc

        return PNPLLibriBrain100(
            data_path=str(self.path),
            task=WordClassification(tmin=0.0, tmax=0.5),
            partition=None,
            include_run_keys=[self.run_key],
            standardize=False,
            clipping_boundary=None,
            download=True,
        )

    @property
    def run_key(self) -> tuple[str, str, str, str]:
        return (self.subject, self.session, self.task, self.run)

    def _load_events(self) -> pd.DataFrame:
        ds = self._pnpl_dataset()
        events_path = ds.ensure_file(ds.get_events_path(*self.run_key))
        events = pd.read_csv(events_path, sep="\t")

        h5_path = ds.ensure_file(ds.get_h5_path(*self.run_key))
        with h5py.File(h5_path, "r") as h5:
            sfreq = float(h5.attrs["sample_frequency"])
            duration = h5["data"].shape[1] / sfreq

        words = events.loc[events.kind == "word"].copy()
        words = words.dropna(subset=["segment", "timemeg"])
        words["segment"] = words["segment"].astype(str).str.strip()
        words = words.loc[words.segment != ""]

        out = pd.DataFrame(
            {
                "type": "Word",
                "start": pd.to_numeric(words.timemeg, errors="coerce"),
                "duration": pd.to_numeric(
                    _column_or_default(words, "duration", 0.0), errors="coerce"
                ).fillna(0.0),
                "text": words.segment,
                "language": "english",
                "sentence": _column_or_default(words, "sentence", words.segment),
                "sequence_id": _column_or_default(words, "sentenceidx", 0)
                .fillna(0)
                .astype(int),
                "word_idx": _column_or_default(words, "wordidx", 0)
                .fillna(0)
                .astype(int),
                "split": self.partition.replace("validation", "val"),
                "corpus": self.corpus,
                "task": self.task,
                "session": self.session,
                "run": self.run,
            }
        )

        meg = pd.DataFrame(
            [
                {
                    "type": "Meg",
                    "filepath": f"method:_load_raw?timeline={self.timeline}",
                    "start": 0.0,
                    "duration": duration,
                    "frequency": sfreq,
                    "split": self.partition.replace("validation", "val"),
                    "corpus": self.corpus,
                    "task": self.task,
                    "session": self.session,
                    "run": self.run,
                }
            ]
        )
        return pd.concat([meg, out], ignore_index=True)

    def _load_raw(self, timeline: str) -> mne.io.RawArray:
        # ``timeline`` is included in the URI for cache uniqueness.
        del timeline
        ds = self._pnpl_dataset()
        h5_path = ds.ensure_file(ds.get_h5_path(*self.run_key))
        with h5py.File(h5_path, "r") as h5:
            data = np.asarray(h5["data"], dtype=np.float64)
            sfreq = float(h5.attrs["sample_frequency"])
            ch_names = _decode_attr(h5.attrs.get("channel_names"))
            ch_types = _decode_attr(h5.attrs.get("channel_types"))

        if not ch_names:
            ch_names = [f"MEG{i:03d}" for i in range(data.shape[0])]
        if not ch_types:
            ch_types = ["mag"] * len(ch_names)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        return mne.io.RawArray(data, info, verbose=False)
