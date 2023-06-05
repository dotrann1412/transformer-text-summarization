import api from '../utils/request'

export function summarize(text: string, summarize_ratio: number) {
    return api.post<any>('/cores/summarize', { text: text, summarize_ratio: summarize_ratio })
}