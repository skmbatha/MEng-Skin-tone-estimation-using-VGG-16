PROBLEM

Typically data copied fron the data collected need cleaning. But the cleaning step is easier
done when the data is already grouped into a regression type dataset with both "training" and
"validation" folders. These folders are normally paired with "annotations.csv" files.
During cleaning images can be easily removed, leaving annotations that have no images they are
linked to.


SOLUTION

This scriptlet take a regression folder (specify either the training or validation folder). 
It then steps through every annotation and validates if the image is available, if not the label
is removed. A new annotation file is created and replaces the old one.