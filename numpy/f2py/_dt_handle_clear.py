"""Release owned native storage while retaining the validated handle identity."""


def clear_source(description):
    prefix = description['prefix']
    if description['lengths']:
        declaration = f'type({description["box"]}), pointer :: old_box'
        release = ['call c_f_pointer(saved, old_box)',
                   'if (old_box%owns_value .and. associated(old_box%value)) &',
                   '  deallocate(old_box%value)', 'deallocate(old_box)']
    else:
        declaration = f'type({description["pointer_spec"]}), pointer :: old_value'
        release = ['call c_f_pointer(saved, old_value)',
                   'if (associated(old_value)) deallocate(old_value)']
    lines = [f'integer(c_int) function {prefix}_clear(handle) result(status)',
             'type(c_ptr), value :: handle',
             'type(f2py_pdt_handle), pointer :: header',
             'type(c_ptr) :: saved', declaration, 'status = 4',
             'header => f2py_pdt_header(handle)',
             'if (.not. associated(header)) return',
             f'if (header%identity /= {description["identity_number"]}_c_int64_t) return',
             'if (header%storage /= 0 .or. header%writable == 0) return',
             'saved = header%payload', 'header%payload = c_null_ptr',
             'if (c_associated(saved)) then', *release, 'end if', 'status = 0',
             f'end function {prefix}_clear', '']
    return '\n'.join(lines)
