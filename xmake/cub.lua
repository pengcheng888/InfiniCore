package("infinicub")
    set_description("Build infinicub library.")
    
    local dir = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
    set_installdir(path.join(dir,"packages"))
    set_sourcedir(path.join(os.scriptdir(), "../src/infinicub"))
    
    add_configs("shared", {default = false, type = "boolean", readonly = true})
    on_install(function (package)
        local configs = {}
        import("package.tools.xmake").install(package, configs)
    end)
package_end()
