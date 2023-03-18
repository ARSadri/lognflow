History
=======

0.1.0 (2022-11-16)
------------------

* First release on PyPI.

0.3.0 (2022-12-19)
------------------

* Very consistent and easy-to-handle interface

0.3.2 (2022-12-27)
------------------

* log your dictionary.

0.5.2 (2023-01-11)
------------------

* logger call takes txt only for the main_log
* main_log name can have no time_stamp
* All logs can be force_flush = True.
* __del__ is well implemented.

0.5.3 (2023-03-19)
------------------

* use logger.log_single('aName', aDict) to save a dictionary using numpy.savez
* time_in_file_name has been changed to time_tag, sorry there! but early stages.
* time_tag is True in the constructor, but could always be set to False
* All tests passed!