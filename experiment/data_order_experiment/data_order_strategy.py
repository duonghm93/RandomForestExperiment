def do_nothing_strategy(df):
    return df


def sort_by_slot_id_strategy(df):
    return _sort_strategy(df, 'slot_id')


def sort_by_dow_strategy(df):
    return _sort_strategy(df, 'dow')


def sort_by_isclick_strategy(df):
    return _sort_strategy(df, 'is_click')


def _sort_strategy(df, col_to_sort):
    return df.sort_values(col_to_sort)


def _sort_strategy_generator(df):
    generate_strategies = []
    generate_strategies.append(('do_nothing', lambda df: df))

    # columns = df.columns
    # for col_name in columns:
    for col_name in ['slot_id', 'dow', 'is_click']:
        strategy_name = 'sort_by_' + col_name
        # def data_sort(col_to_sort):
        #     return df.sort_values(col_to_sort)
        func = (lambda df: df.sort_values(col_name))
        generate_strategies.append((strategy_name, func))
    return generate_strategies


strategies = {
    'do_nothing': do_nothing_strategy,
    'sort_by_slot_id': sort_by_slot_id_strategy,
    'sort_by_dow_strategy': sort_by_dow_strategy,
    'sort_by_is_click': sort_by_isclick_strategy
}

