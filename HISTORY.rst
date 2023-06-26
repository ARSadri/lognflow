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
* All text files are handled by the with statement
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

0.6.4 (2023-04-06)
------------------
* Better documentation and examples for readme
* get_var is added to lognflow to get buffered variables logged by log_var

0.6.5 (2023-04-26)
------------------
* Fixed a bug in the docs to allow sphinx compile it.
* log_var will log only the valid time stamps.
* added end keyword argument to log_text

0.6.6 (2023-04-27)
------------------
* Better documentation
* added tifffile imread to logviewer and imwrite to lognflow

0.6.7 (2023-04-27)
------------------
* A bug in tifffile support was fixed

0.6.8 (2023-05-04)
------------------
* Fixing readme for PyPI.
* removed marker from log_plot. user marker and linestyle keyword arguments.
* printprogress returns proper ETA every time if print_function is set to None::

0.7.0 (2023-05-15)
------------------
* logviewer returns data by log_sigle if the full name is mentioned.

0.7.1 (2023-05-22)
------------------
* printprogress supports lognflow.
* bugs fixed in lognflow.
* For now I guess lognflow and logviewer could be separate.

0.7.2 (2023-05-25)
------------------
* bug fixed in logviewer
* text_to_object added to logviewer to read dict or list logged via log_single
* test pass for logviewer including the test for text_to_object

0.7.3 (2023-06-01)
------------------
* bug fixed in logviewer in the use of suffix in get_stack_of_files
* log_imshow takes colorbar and remove_axis_ticks flags.
* every lognflow instance has a logviewer pointing to its log_dir called logged.

0.7.4 (2023-06-26)
------------------
* critical bug fixed in log_imshow