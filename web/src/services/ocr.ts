import api from '../utils/request'

export function upload(file: any) {
    return api.post<any>('/cores/upload/', {
        filename: file,
    })
}