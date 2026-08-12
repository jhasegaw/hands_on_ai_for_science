# ----------------------------------------------------------------------------
# UNVERIFIED SOLUTION
#
# Written for the workshop -- this is NOT the course author's original.
# It reproduces the expected output published in the notebook, but it has not
# been reviewed by whoever designed the assignment.  Treat it as a reference
# for a TA, not as an answer key.  See solutions/README.md.
# ----------------------------------------------------------------------------
def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters

    Every element of "words" should be converted to a str, then split into
    characters, each of which is separately appended to "characters." For
    example, if words==['hello', 1.234, True], then characters should be
    ['h', 'e', 'l', 'l', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    """
    # Start with an empty list and add to it as we go.  This is the standard
    # "accumulator" pattern: make an empty container, then fill it in a loop.
    characters = []

    for word in words:
        # The elements are not always strings -- the example includes the
        # number 1.234 and the boolean True.  str() turns any of them into
        # text, so that 1.234 becomes "1.234" and True becomes "True".
        text = str(word)

        # Looping over a string visits it one character at a time, so this
        # inner loop appends 'h', then 'e', then 'l', and so on.
        for character in text:
            characters.append(character)

    return characters


def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    # Sorting the dates puts them in calendar order.  Python compares tuples
    # element by element, so (2,12) < (2,22) < (4,13): it checks the month
    # first and only looks at the day when the months are equal.
    sorted_dates = sorted(birthdays.keys())

    # Walk through the calendar in order and stop at the first date that is
    # strictly later than the one we were given.
    for birthday in sorted_dates:
        if birthday > date:
            return birthday, birthdays[birthday]

    # If we get here, no birthday falls later in the year -- for example, the
    # given date is December 26 and the last birthday is April 13.  The next
    # birthday is then the earliest one, in January of the following year.
    # This wrap-around is the case that is easy to forget.
    earliest = sorted_dates[0]
    return earliest, birthdays[earliest]
