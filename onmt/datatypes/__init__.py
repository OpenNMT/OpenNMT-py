"""Module defining data types.

Datatypes implement most of the logic of the different
common inputs to sequence models. They are designed to make
adding new datatypes usually mean only adding a new datatype file.
A library user can add a new datatype without altering
onmt by adding their datatype to the str2datatype dict here.
"""

from onmt.datatypes.audio_datatype import audio
from onmt.datatypes.text_datatype import text
from onmt.datatypes.image_datatype import image
from onmt.datatypes.datareader_base import DataReaderBase
from onmt.datatypes.datatype_base import Datatype


str2datatype = {
    audio.name: audio,
    text.name: text,
    image.name: image
}


__all__ = ['DataReaderBase', 'Datatype', 'audio', 'text', 'image',
           'str2datatype']
