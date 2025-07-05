package("infinicub")
    set_sourcedir(path.join(os.scriptdir(), "../src/infinicub"))
    
    add_configs("shared", {description = "Build shared library.", default = false, type = "boolean", readonly = true})

    on_install(function (package)
        -- add_configs("feature", {description = "Enable feature", default = false, type = "boolean"})
        local configs = {}
        if package:config("shared") then
            configs.kind = "shared"
        end
        import("package.tools.xmake").install(package, configs)
    end)
package_end()