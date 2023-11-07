"""
Create a macro for SAXS to scan direct beam images.

call with "python make_SAXS_macro.py [start] [finish] [step] [file-comment]"
where in [] are values (don't type [])

author: Teddy Tortorici
"""
import os
import datetime


def create_file(angles: list[float], tag: str = "") -> None:
    """
    Create a macro file for scanning angle for GIWAXS
    :param angles: list of angles to scan through
    :param tag: optional identifier to put in filename
    :return: None
    """
    print(angles)
    if tag:
        tag += "_"
    date = datetime.datetime.now()
    macroname = f'GIWAXS_tune_{date.year}-{date.month}-{date.day}.txt'
    print("Writing Macro...")
    if macroname[-4:].lower() != ".txt":
        macroname = macroname.replace(".", "") + ".txt"
    with open(os.path.join("Macros", macroname), 'w') as f:
        f.write("umvr wbs -5\n")  # move beam stop out of the way
        f.write("umvr z -10\n")  # move sample out of the way
        f.write("eiger_run 0.1 direct_beam.tif\n")  # take direct beam exposure
        f.write("umvr z 10\n")  # move sample back into beam

        for om in angles:
            f.write(f"umv om {om}\n")
            formatted_angle = "{}_{}".format(*str(om).split("."))
            f.write(f"eiger_run 0.1 db_{tag}{formatted_angle}_degrees.tif\n")

        f.write("umvr wbs 5\n")
        f.write("umv om 0\n")
    print("Macro written")
    print("Copy and paste the following into SAXS to run the macro:")
    print("do " + os.path.join(os.getcwd(), macroname))
    return None


def arange(start, finish, step):
    """
    Make a list of values similar to np.arange, but with values rounded to avoid floating point precision issues
    :param start: first element of the list
    :param finish: last element of the list
    :param step: difference between sequential elements
    :return: list of values
    """
    # Try to determine what digit to round to
    step_decimal = str(float(step)).split(".")  # list[str]: [left of decimal, right of decimal]
    if step_decimal[1] == 0:        # then step is an integer
        rounding = 0
        # find lowest order non-zero digit
        while True:
            if step_decimal[0][::-1].find('0') == 0:
                step_decimal[0] = step_decimal[0][:-1]
                rounding -= 1
            else:
                break
    else:                           # then step is not an integer
        rounding = len(step_decimal[1])     # number of digits right of the decimal
    return [round(x * step + start, rounding) for x in list(range(int((finish + step - start) / step)))]


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # sys.argv += [-0.8, 0.8, 0.05, ""]
        sys.argv += [0.0, 1.2, 0.005, ""]
    elif len(sys.argv) == 4:
        sys.argv += [""]

    create_file(angles=arange(start=float(sys.argv[1]), finish=float(sys.argv[2]), step=float(sys.argv[3])),
                tag=sys.argv[4])
    print("okay")
