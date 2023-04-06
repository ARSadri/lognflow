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

0.6.0 (2023-03-22)
------------------
* This is a stable release
* All text files are handled by thw with statement
* Renaming bug is fixed
* all tests run properly.
* lognflow has all that logviewer has. We will check if dir exists at every use

0.6.1 (2023-03-22)
------------------
* rename had a bug that is fixed

0.6.2 (2023-03-25)
------------------
* made it possible to flush_all()
* We support Python 3.7+ because of dataclasses
* printprogress now can disable printing anything and return ETA at the __call__

0.6.3 (2023-04-01)
------------------
* lognflow class does all logviewer does. Maybe it is time to remove logviewer.

0.6.4 (2023-04-10)
------------------
* Better documentation and examples for readme
* get_var is added to lognflow