"""
db/codegen.py
Reads schema.prisma from project root, writes db/schema.py.

Run from project root:
    python -m db.codegen

Workflow:
    1. Edit schema.prisma
    2. python -m db.codegen      -> regenerates db/schema.py
    3. python -m db.migrations   -> applies changes to Neon
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Some table names are referenced by a shorter key in the rest of the codebase.
# e.g. sheets.py uses read_tab("geo_data") but the actual table is geopolitical_data.
# Add overrides here: { "actual_table_name": "key_used_in_code" }
# ---------------------------------------------------------------------------
TAB_KEY_OVERRIDES = {
    "geopolitical_data": "geo_data",
    "db_schema":         "schema",
}


def parse_prisma(text: str) -> list:
    """
    Parse all model blocks from schema.prisma.
    Returns list of dicts:
        {
            name:    str,
            fields:  list of {name, required, is_serial, is_pk, default},
            uniques: list of list[str],
            indexes: list of list[str],
        }
    """
    # Strip inline // comments before parsing
    lines = []
    for line in text.splitlines():
        line = re.sub(r'\s*//.*$', '', line)
        lines.append(line)
    clean = '\n'.join(lines)

    models = []
    for m in re.finditer(r'model\s+(\w+)\s*\{([^}]*)\}', clean, re.DOTALL):
        table_name = m.group(1)
        body       = m.group(2)

        fields  = []
        uniques = []
        indexes = []

        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue

            # @@unique([col, col])
            um = re.match(r'@@unique\(\[([^\]]+)\]\)', line)
            if um:
                cols = [c.strip() for c in um.group(1).split(',')]
                uniques.append(cols)
                continue

            # @@index([col, col])
            im = re.match(r'@@index\(\[([^\]]+)\]\)', line)
            if im:
                cols = [c.strip() for c in im.group(1).split(',')]
                indexes.append(cols)
                continue

            # Field line: name  Type  [annotations]
            parts = line.split()
            if len(parts) < 2:
                continue

            field_name = parts[0]
            field_type = parts[1]
            attrs      = ' '.join(parts[2:])

            is_serial = field_type == 'Serial'
            is_pk     = is_serial or '@id' in attrs
            required  = field_type.endswith('!') or is_pk
            has_unique = '@unique' in attrs

            default_m = re.search(r'@default\("([^"]+)"\)', attrs)
            if not default_m:
                default_m = re.search(r"@default\('([^']+)'\)", attrs)
            if not default_m:
                default_m = re.search(r'@default\(([0-9][0-9.]*)\)', attrs)  # unquoted numeric
            default = default_m.group(1) if default_m else None

            # @unique on a field = single-col unique constraint
            if has_unique and not is_serial:
                uniques.append([field_name])

            fields.append({
                'name':      field_name,
                'required':  required,
                'is_serial': is_serial,
                'is_pk':     is_pk,
                'default':   default,
            })

        models.append({
            'name':    table_name,
            'fields':  fields,
            'uniques': uniques,
            'indexes': indexes,
        })

    return models


def model_to_ddl(model: dict) -> str:
    """Generate CREATE TABLE IF NOT EXISTS SQL for one model."""
    name   = model['name']
    fields = model['fields']

    col_lines = []

    for f in fields:
        col = f['name']
        if f['is_serial']:
            col_lines.append(f'        {col} SERIAL PRIMARY KEY')
            continue
        if f['is_pk'] and not f['is_serial']:
            col_lines.append(f'        {col} TEXT NOT NULL PRIMARY KEY')
            continue
        null_part    = ' NOT NULL' if f['required'] else ''
        default_part = f" DEFAULT '{f['default']}'" if f['default'] else ''
        col_lines.append(f'        {col} TEXT{null_part}{default_part}')

    # Only auto-add inserted_at if the model doesn't already define it explicitly
    field_names = {f['name'] for f in fields}
    if 'inserted_at' not in field_names:
        col_lines.append('        inserted_at TIMESTAMPTZ DEFAULT NOW()')

    # Inline UNIQUE constraints (skip if already primary key)
    pk_names = {f['name'] for f in fields if f['is_pk']}
    for cols in model['uniques']:
        if len(cols) == 1 and cols[0] in pk_names:
            continue
        col_str  = ', '.join(cols)
        idx_name = 'ux_' + name + '_' + '_'.join(cols)
        col_lines.append(
            f'        CONSTRAINT {idx_name} UNIQUE ({col_str})'
        )

    cols_sql = ',\n'.join(col_lines)
    ddl = f'    CREATE TABLE IF NOT EXISTS "{name}" (\n{cols_sql}\n    );\n'

    # Separate CREATE INDEX statements
    for cols in model['indexes']:
        col_str  = ', '.join(cols)
        idx_name = 'ix_' + name + '_' + '_'.join(cols)
        ddl += (
            f'    CREATE INDEX IF NOT EXISTS {idx_name}\n'
            f'        ON "{name}" ({col_str});\n'
        )

    return ddl


def model_to_columns(model: dict) -> list:
    """Column list for TABLE_COLUMNS — excludes Serial id and inserted_at."""
    return [f['name'] for f in model['fields'] if not f['is_serial']]


def generate_schema_py(models: list) -> str:
    """Produce the full db/schema.py content."""

    # TABS
    tab_lines = []
    for m in models:
        key = TAB_KEY_OVERRIDES.get(m['name'], m['name'])
        tab_lines.append(f'    "{key}": "{m["name"]}"')
    tabs_str = ',\n'.join(tab_lines)

    # TABLE_DDL
    ddl_parts = []
    for m in models:
        ddl = model_to_ddl(m)
        ddl_parts.append(f'    "{m["name"]}": """\n{ddl}    """')
    ddl_str = ',\n\n'.join(ddl_parts)

    # TABLE_COLUMNS
    col_parts = []
    for m in models:
        cols     = model_to_columns(m)
        cols_str = ', '.join(f'"{c}"' for c in cols)
        col_parts.append(f'    "{m["name"]}": [{cols_str}]')
    cols_str_all = ',\n'.join(col_parts)

    return (
        '"""\n'
        'db/schema.py\n'
        'AUTO-GENERATED by db/codegen.py — DO NOT EDIT MANUALLY.\n'
        'Edit schema.prisma then run: python -m db.codegen\n'
        '"""\n\n'
        'TABS: dict[str, str] = {\n'
        + tabs_str + ',\n'
        '}\n\n'
        'TABLE_DDL: dict[str, str] = {\n'
        + ddl_str + ',\n'
        '}\n\n'
        'TABLE_COLUMNS: dict[str, list[str]] = {\n'
        + cols_str_all + ',\n'
        '}\n'
    )


def main():
    root        = Path(__file__).parent.parent
    prisma_path = root / 'schema.prisma'
    output_path = root / 'db' / 'schema.py'

    if not prisma_path.exists():
        print(f'ERROR: schema.prisma not found at {prisma_path}')
        return

    text   = prisma_path.read_text(encoding='utf-8')
    models = parse_prisma(text)

    if not models:
        print('ERROR: No models found in schema.prisma')
        return

    content = generate_schema_py(models)
    output_path.write_text(content, encoding='utf-8')

    print(f'Generated db/schema.py from {len(models)} models:')
    for m in models:
        cols = model_to_columns(m)
        print(f'  {m["name"]:35s} {len(cols)} columns')
    print(f'\nOutput: {output_path}')


if __name__ == '__main__':
    main()
