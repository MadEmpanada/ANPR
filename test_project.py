from project import arguments
import pytest
from project import get_car_detections
from project import output_detection

#Tests required command-line arguments
def test_arguments():
    with pytest.raises(SystemExit) as error:
        arguments()
    assert error.value.code == 2

#Tests detections with correct file path
def test_get_car_detections():
    assert isinstance(get_car_detections(video_path="sample1_30fps.mp4"), list) == True

#Tests boolean output regardless of the argument values
def test_output_detection():
    return isinstance(output_detection(images=[], width=0, height=0), bool) == True
