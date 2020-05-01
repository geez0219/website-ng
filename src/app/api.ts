export interface API {
    name: string;
    displayName: string;
    sourceurl: string;
    type: string;
    children?: API[];
}
