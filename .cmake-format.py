# dlaf-no-license-check
# ----------------------------------
# Options affecting listfile parsing
# ----------------------------------
with section("parse"):

  # Specify structure for custom cmake functions
  additional_commands = {
    "DLAF_addTest": {
      "pargs": 1,
      "flags": [],
      "kwargs": {
        "SOURCES": '*',
        "COMPILE_DEFINITIONS": '*',
        "INCLUDE_DIRS": '*',
        "LIBRARIES": '*',
        "ARGUMENTS": '*',
        "USE_MAIN": '1',
        "MPIRANKS": '1',
      }
    },
    "DLAF_addMiniapp": {
      "pargs": 1,
      "flags": [],
      "kwargs": {
        "SOURCES": '*',
        "COMPILE_DEFINITIONS": '*',
        "INCLUDE_DIRS": '*',
        "LIBRARIES": '*',
        "ARGUMENTS": '*',
        "USE_MAIN": '1',
        "MPIRANKS": '1',
      }
    },
    "DLAF_addSublibrary": {
      "pargs": 1,
      "flags": [],
      "kwargs": {
        "SOURCES": '*',
        "GPU_SOURCES": '*',
        "COMPILE_OPTIONS": '*',
        "LIBRARIES": '*',
      }
    },
    "DLAF_addPrecompiledHeaders": {
      "pargs": 1,
    },
    "target_add_warnings": {
      "pargs": 1,
    }
  }

  # Override configurations per-command where available
  override_spec = {}

  # Specify variable tags.
  vartags = []

  # Specify property tags.
  proptags = []

# -----------------------------
# Options affecting formatting.
# -----------------------------
with section("format"):

  # Disable formatting entirely, making cmake-format a no-op
  disable = False

  # How wide to allow formatted cmake files
  line_width = 105

  # How many spaces to tab for indent
  tab_size = 2

  # If true, lines are indented using tab characters (utf-8 0x09) instead of
  # <tab_size> space characters (utf-8 0x20). In cases where the layout would
  # require a fractional tab character, the behavior of the  fractional
  # indentation is governed by <fractional_tab_policy>
  use_tabchars = False

  # If an argument group contains more than this many sub-groups (parg or kwarg
  # groups) then force it to a vertical layout.
  max_subgroups_hwrap = 3

  # If a positional argument group contains more than this many arguments, then
  # force it to a vertical layout.
  max_pargs_hwrap = 6

  # If a cmdline positional group consumes more than this many lines without
  # nesting, then invalidate the layout (and nest)
  max_rows_cmdline = 2

  # If true, separate flow control names from their parentheses with a space
  separate_ctrl_name_with_space = False

  # If true, separate function names from parentheses with a space
  separate_fn_name_with_space = False

  # If a statement is wrapped to more than one line, than dangle the closing
  # parenthesis on its own line.
  dangle_parens = True

  # If the trailing parenthesis must be 'dangled' on its on line, then align it
  # to this reference: `prefix`: the start of the statement,  `prefix-indent`:
  # the start of the statement, plus one indentation  level, `child`: align to
  # the column of the arguments
  dangle_align = 'prefix'

  # If the statement spelling length (including space and parenthesis) is
  # smaller than this amount, then force reject nested layouts.
  min_prefix_chars = 4

  # If the statement spelling length (including space and parenthesis) is larger
  # than the tab width by more than this amount, then force reject un-nested
  # layouts.
  max_prefix_chars = 10

  # If a candidate layout is wrapped horizontally but it exceeds this many
  # lines, then reject the layout.
  max_lines_hwrap = 2

  # What style line endings to use in the output.
  line_ending = 'unix'

  # Format command names consistently as 'lower' or 'upper' case
  command_case = 'canonical'

  # Format keywords consistently as 'lower' or 'upper' case
  keyword_case = 'unchanged'

  # A list of command names which should always be wrapped
  always_wrap = []

  # If true, the argument lists which are known to be sortable will be sorted
  # lexicographicall
  enable_sort = True

  # If true, the parsers may infer whether or not an argument list is sortable
  # (without annotation).
  autosort = False

  # By default, if cmake-format cannot successfully fit everything into the
  # desired linewidth it will apply the last, most agressive attempt that it
  # made. If this flag is True, however, cmake-format will print error, exit
  # with non-zero status code, and write-out nothing
  require_valid_layout = False

  # A dictionary mapping layout nodes to a list of wrap decisions. See the
  # documentation for more information.
  layout_passes = {}

# ------------------------------------------------
# Options affecting comment reflow and formatting.
# ------------------------------------------------
with section("markup"):

  # enable comment markup parsing and reflow
  enable_markup = False

# ----------------------------
# Options affecting the linter
# ----------------------------
with section("lint"):

  # a list of lint codes to disable
  disabled_codes = []

  # regular expression pattern describing valid function names
  function_pattern = '[0-9a-z_]+'

  # regular expression pattern describing valid macro names
  macro_pattern = '[0-9A-Z_]+'

  # regular expression pattern describing valid names for variables with global
  # (cache) scope
  global_var_pattern = '[A-Z][0-9A-Z_]+'

  # regular expression pattern describing valid names for variables with global
  # scope (but internal semantic)
  internal_var_pattern = '_[A-Z][0-9A-Z_]+'

  # regular expression pattern describing valid names for variables with local
  # scope
  local_var_pattern = '[a-z][a-z0-9_]+'

  # regular expression pattern describing valid names for privatedirectory
  # variables
  private_var_pattern = '_[0-9a-z_]+'

  # regular expression pattern describing valid names for public directory
  # variables
  public_var_pattern = '[A-Z][0-9A-Z_]+'

  # regular expression pattern describing valid names for function/macro
  # arguments and loop variables.
  argument_var_pattern = '[a-z][a-z0-9_]+'

  # regular expression pattern describing valid names for keywords used in
  # functions or macros
  keyword_pattern = '[A-Z][0-9A-Z_]+'

  # In the heuristic for C0201, how many conditionals to match within a loop in
  # before considering the loop a parser.
  max_conditionals_custom_parser = 2

  # Require at least this many newlines between statements
  min_statement_spacing = 1

  # Require no more than this many newlines between statements
  max_statement_spacing = 2
  max_returns = 6
  max_branches = 12
  max_arguments = 5
  max_localvars = 15
  max_statements = 50

# -------------------------------
# Options affecting file encoding
# -------------------------------
with section("encode"):

  # If true, emit the unicode byte-order mark (BOM) at the start of the file
  emit_byteorder_mark = False

  # Specify the encoding of the input file. Defaults to utf-8
  input_encoding = 'utf-8'

  # Specify the encoding of the output file. Defaults to utf-8. Note that cmake
  # only claims to support utf-8 so be careful when using anything else
  output_encoding = 'utf-8'

# -------------------------------------
# Miscellaneous configurations options.
# -------------------------------------
with section("misc"):

  # A dictionary containing any per-command configuration overrides. Currently
  # only `command_case` is supported.
  per_command = {}
