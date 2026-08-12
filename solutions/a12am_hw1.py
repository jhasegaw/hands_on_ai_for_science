def arithmetic(x, y):
    """
    Modify this code so that it performs one of four possible functions,
    as specified in the following table:

                        isinstance(x,str)  isinstance(x,float)
    isinstance(y,str)   return x+y         return str(x)+y
    isinstance(y,float) return x*int(y)    return x*y
    """
    # The table has two rows and two columns, so there are four cases.  Which
    # case we are in depends on the TYPE of each argument, not on its value,
    # so every branch below is an isinstance test.
    #
    # Read the table one row at a time: the row picks what kind of thing y is,
    # and the column picks what kind of thing x is.

    if isinstance(y, str):
        # Top row of the table: y is a string, so the answer is a string.
        if isinstance(x, str):
            # Two strings glue together: "very" + "cool" -> "verycool"
            return x + y
        else:
            # x is a number.  Python will not glue a number onto a string, so
            # convert it first: str(3.0) + "times" -> "3.0times"
            return str(x) + y
    else:
        # Bottom row of the table: y is a number, so it is a repeat count or a
        # multiplier.
        if isinstance(x, str):
            # Repeating a string needs a whole number of copies, and y may be
            # a float like 3.0, so int() it: "times" * int(3.0) -> "timestimestimes"
            return x * int(y)
        else:
            # Two numbers: ordinary multiplication.  2.0 * 5.0 -> 10.0
            return x * y
