import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'; 

import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTreeNestedDataSource } from '@angular/material/tree';

import { API } from '../api';

@Component({
  selector: 'app-api',
  templateUrl: './api.component.html',
  styleUrls: ['./api.component.css']
})
export class ApiComponent implements OnInit {
  apiList: API[];
  selectedAPI: string;
  currentSelection: string;
  currentAPIText: string;

  treeControl: NestedTreeControl<API>;
  dataSource: MatTreeNestedDataSource<API>;

  constructor(private http: HttpClient) { }

  ngOnInit() {
    this.treeControl = new NestedTreeControl<API>(node => node.children);
    this.dataSource = new MatTreeNestedDataSource<API>();

    this.getAPIStructure();
    this.selectedAPI = "fe/Pipeline.md";
    this.currentSelection = 'assets/api/fe/Pipeline.md';
    this.getSelectedAPIText();
  }

  getAPIStructure() {
    this.http.get('assets/api/structure.json', {responseType: 'text'}).subscribe(data => {
      this.apiList = <API[]>JSON.parse(data);

      this.dataSource.data = this.apiList;

      this.treeControl.dataNodes = this.apiList;
      this.treeControl.expand(this.treeControl.dataNodes[0]);
    });
  }
  
  hasChild = (_: number, node: API) => !!node.children && node.children.length > 0;

  updateCurrentAPI(api: API) {
    window.scroll(0,0);
    this.selectedAPI = api.name;
    this.currentSelection = 'assets/api/' + api.name;

    this.getSelectedAPIText();
  }

  getSelectedAPIText() {
    this.http.get(this.currentSelection, {responseType: 'text'}).subscribe(data => {
      this.currentAPIText = data;
    });
  }

}
