import api from '../utils/request'

export function summarize(text: string) {
    return api.post<any>('/cores/summarize/', {
        text: text,
    })
}