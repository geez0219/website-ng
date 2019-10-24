export interface API {
    name: string;
    displayName: string;
    type: string;
    children?: API[];
}
