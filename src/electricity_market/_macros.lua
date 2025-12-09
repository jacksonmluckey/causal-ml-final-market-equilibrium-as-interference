function Pandoc(doc)
  quarto.log.warning('=== FILTER RUNNING ===')

  local file = io.open('_macros.tex', 'r')
  if not file then 
    quarto.log.error('Cannot open _macros.tex')
    return doc 
  end

  local content = file:read('*a')
  file:close()

  quarto.log.warning('File read, content length: ' .. #content)

  if quarto.doc.is_format('html') then
    quarto.log.warning('HTML format detected!')

    local macros_json = "{"
    local first = true
    local count = 0

    local i = 1
    while i <= #content do
      local start = content:find('\\newcommand{\\', i)
      if not start then break end

      local name_start = start + 13
      local name_end = content:find('}', name_start)
      local name = content:sub(name_start, name_end - 1)

      local def_start = name_end + 2
      local brace_count = 1
      local def_end = def_start

      while def_end <= #content and brace_count > 0 do
        local char = content:sub(def_end, def_end)
        if char == '{' then brace_count = brace_count + 1
        elseif char == '}' then brace_count = brace_count - 1
        end
        if brace_count > 0 then def_end = def_end + 1 end
      end

      local def = content:sub(def_start, def_end - 1)
      count = count + 1

      if not first then macros_json = macros_json .. "," end
      first = false

      def = def:gsub('\\', '\\\\'):gsub('"', '\\"')
      macros_json = macros_json .. '"' .. name .. '":"' .. def .. '"'

      i = def_end + 1
    end

    macros_json = macros_json .. "}"
    quarto.log.warning('Found ' .. count .. ' macros')

    -- Append directly to document body instead of metadata
    local script = '<script>\n' ..
      'window.MathJax = { tex: { macros: ' .. macros_json .. ' } };\n' ..
      '</script>'

    table.insert(doc.blocks, pandoc.RawBlock('html', script))
    quarto.log.warning('Script appended to doc.blocks')
  end

  return doc
end