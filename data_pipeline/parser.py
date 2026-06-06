"""Base parser interface for the data pipeline.

All parsers consume a decoded tensor dictionary and return (image, labels).
Training and evaluation paths are separated via _parse_train_data /
_parse_eval_data so subclasses only need to override the relevant method.

Classes:
    Parser: Abstract base class.
"""

import abc
from typing import Dict, Tuple

import tensorflow as tf


class Parser(abc.ABC):
    """Abstract base class for data parsers."""

    def parse_fn(self, is_training: bool):
        """Return a callable that dispatches to train or eval parsing."""

        def _parse(data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict]:
            if is_training:
                return self._parse_train_data(data)
            return self._parse_eval_data(data)

        return _parse

    @abc.abstractmethod
    def _parse_train_data(
        self, data: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict]:
        """Parse and augment a single training example."""

    @abc.abstractmethod
    def _parse_eval_data(
        self, data: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict]:
        """Parse a single evaluation example (minimal augmentation)."""
